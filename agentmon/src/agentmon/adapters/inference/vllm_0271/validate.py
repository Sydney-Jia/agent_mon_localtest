"""Exact-ID event-chain validation. Never infer IDs from suffixes or timestamps."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .contracts import KV_FILE_SHA256


def load_validation_archive(directory, *, require_live=False, health_directory=None):
    """Read and check archive integrity once; reuse for exact-ID validations."""
    directory = Path(directory)
    records = []
    errors = []
    files = sorted(directory.glob('hook-*.jsonl'))
    for path in files:
        previous_seq = previous_time = 0
        previous_pid = previous_epoch = None
        for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            try:
                event = json.loads(line)
                required = ('event', 'pid', 'engine_epoch', 'run_id', 'seq_no', 'utc_ns',
                            'monotonic_ns', 'source_function', 'evidence_status', 'schema_version')
                if not all(key in event for key in required):
                    raise ValueError('required fields missing')
                if event['seq_no'] <= previous_seq or event['monotonic_ns'] < previous_time:
                    raise ValueError('non-monotonic event order')
                if previous_pid is not None and (event['pid'], event['engine_epoch']) != (previous_pid, previous_epoch):
                    raise ValueError('mixed process/epoch in one event file')
                previous_seq, previous_time = event['seq_no'], event['monotonic_ns']
                previous_pid, previous_epoch = event['pid'], event['engine_epoch']
                if event.get('blocks_truncated'):
                    errors.append('truncated block view')
                if event['event'] in {'HOOK_MODULE_REJECTED', 'ORIGINAL_CALL_EXCEPTION'}:
                    errors.append(event['event'])
                if require_live and event['evidence_status'] != 'VERIFIED':
                    errors.append('non-live provenance')
                records.append(event)
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(f'{path.name}:{line_number}: {type(exc).__name__}')
    if not files:
        errors.append('no event files')
    if len({event['run_id'] for event in records}) != 1:
        errors.append('run identity is missing or mixed')
    health_files = sorted(Path(health_directory or directory).glob('health-*.json'))
    streams=defaultdict(list)
    for row in records:streams[(row['run_id'],row['pid'],row['engine_epoch'])].append(row['seq_no'])
    covered=set()
    if not health_files:
        errors.append('no logger health evidence')
    for path in health_files:
        try:
            health = json.loads(path.read_text())
            scope=(health['run_id'],health['pid'],health['engine_epoch']);covered.add(scope)
            numbers=sorted(streams.get(scope,[]))
            if (health.get('last_seq_no')!=len(numbers) or
                    any(n!=i for i,n in enumerate(numbers,1))):
                errors.append('archive sequence gaps or duplicates: '+path.name)
            if health.get('written_events')!=len(numbers):
                errors.append('archive rows/health mismatch: '+path.name)
            if health['dropped_events'] or health['logger_failures']:
                errors.append('logger drops or failures: '+path.name)
            if health.get('observer_failures', 0):
                errors.append('metadata read failures: '+path.name)
            # Live checks can run while the writer is active. All queued events
            # must have reached disk for an auditable complete chain.
            if health['accepted_events'] != health['written_events']:
                errors.append('logger still draining: '+path.name)
        except (OSError,ValueError, KeyError,TypeError):
            errors.append('invalid logger health: '+path.name)
    if set(streams)-covered:errors.append('event process has no matching logger health')
    return {'records':records,'errors':errors,'require_live':require_live}


def validate_directory(directory, *, api_request_id=None, require_live=False, health_directory=None):
    archive = load_validation_archive(directory, require_live=require_live, health_directory=health_directory)
    return validate_archive(archive, api_request_id=api_request_id)


def validate_archive(archive, *, api_request_id=None):
    """Same validation semantics as validate_directory, with no repeated I/O."""
    records, errors = archive['records'], list(archive['errors'])
    require_live = archive['require_live']
    links = [r for r in records if r['event'] == 'API_REQUEST_LINKED'
             and r.get('api_request_id') is not None
             and (api_request_id is None or r['api_request_id'] == api_request_id)]
    if not links:
        # The response ID may exactly equal the observed external field. This
        # equality is allowed; prefixes/suffixes/time-based similarities are not.
        links = [r for r in records if r['event'] == 'REQUEST_ID_BOUND'
                 and api_request_id is not None and r.get('external_request_id') == api_request_id]
    chains = [];seen_chains=set()
    for link in links:
        request_id = link['request_id']
        events = [r for r in records if r['request_id'] == request_id and r['run_id'] == link['run_id']]
        bound = [r for r in events if r['event'] == 'REQUEST_ID_BOUND' and r.get('external_request_id')]
        engine = [r for r in events if r['event'] == 'ENGINE_REQUEST_ADDED']
        for admission in engine:
            pid, epoch = admission['pid'], admission['engine_epoch']
            chain_key=(link['run_id'],pid,epoch,request_id)
            if chain_key in seen_chains:continue
            seen_chains.add(chain_key)
            local = [r for r in events if (r['pid'], r['engine_epoch']) == (pid, epoch)]
            assigned = [r for r in local if r['event'] == 'KV_BLOCKS_ASSIGNED' and r.get('block_ids')]
            lookup = [r for r in local if r['event'] == 'KV_CACHE_LOOKUP']
            detached = [r for r in local if r['event'] == 'KV_REQUEST_DETACHED_OR_FREED']
            finished = [r for r in local if r['event'] == 'REQUEST_FINISHED']
            started = [r for r in records if (r['pid'], r['engine_epoch']) == (pid, epoch)
                       and r['event'] == 'HOOK_STARTED' and r['process_role'] == 'EngineCore']
            ordered = (assigned and detached and finished and
                       admission['seq_no'] < min(r['seq_no'] for r in assigned) <=
                       max(r['seq_no'] for r in detached) < min(r['seq_no'] for r in finished))
            proven_kv = [r for r in records if (r['pid'], r['engine_epoch']) == (pid, epoch)
                         and r['event'] == 'HOOK_STARTED'
                         and r.get('loaded_target', {}).get('file_sha256') == KV_FILE_SHA256
                         and r.get('loaded_target', {}).get('loaded_code_matches_pinned_source') is True]
            real_core = any(r['source_function'] == 'vllm.v1.engine.core.EngineCore.add_request' for r in started)
            identified_engine_process = any('EngineCore' in (r.get('runtime_process_identity', {}).get('linux_comm') or '')
                                            or 'EngineCore' in r.get('runtime_process_identity', {}).get('multiprocessing_name', '')
                                            for r in started)
            binding_consistent = all(r.get('external_request_id') == bound[0]['external_request_id'] for r in bound) if bound else False
            okay = bool(bound and binding_consistent and started and lookup and ordered)
            if require_live:
                okay = okay and bool(proven_kv and real_core and identified_engine_process and api_request_id)
            chains.append({'request_id': request_id, 'api_request_id': api_request_id or link.get('api_request_id'),
                           'external_request_id': bound[0]['external_request_id'] if bound else None,
                           'engine_pid': pid, 'engine_epoch': epoch,
                           'assigned_groups': sorted({r['group_id'] for r in assigned}),
                           'passed': okay})
    if not chains or not any(chain['passed'] for chain in chains):
        errors.append('no complete exact-ID API→Engine→KV→detach→terminal chain')
    passed = not errors
    return {'passed': passed, 'status': 'LIVE_LOGICAL_HOOK_VERIFIED' if passed and require_live else 'LOCAL_HOOK_PASS' if passed else 'FAIL',
            'scope': 'LIVE_RUNTIME_CHAIN' if require_live else 'LOCAL_EVENT_CHAIN',
            'event_count': len(records), 'process_count': len({r['pid'] for r in records}),
            'chains': chains, 'errors': sorted(set(errors)),
            'physical_allocation': 'NOT_OBSERVED', 'block_generation': 'UNRESOLVED',
            'offload': 'NOT_OBSERVED'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events-dir', type=Path, required=True)
    parser.add_argument('--api-request-id')
    parser.add_argument('--require-live', action='store_true')
    args = parser.parse_args(argv)
    result = validate_directory(args.events_dir, api_request_id=args.api_request_id, require_live=args.require_live)
    print(json.dumps(result, indent=2))
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
