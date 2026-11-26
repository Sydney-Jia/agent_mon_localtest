"""Offline audit of an existing C1 run. Never contacts the service."""
from __future__ import annotations

import json
from pathlib import Path
from .c1_observations import classify, process_epoch_stable, summarize


def audit_run(run_dir):
    run_dir = Path(run_dir).resolve()
    missing, errors, inputs, rows = [], [], {}, []
    for name in ('response_summary.json', 'runtime_identity.json', 'preflight.json',
                 'metric_deltas.json', 'residency_evidence.json', 'single_request_report.json'):
        path = run_dir/name
        try:
            inputs[name] = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            missing.append(name)
            errors.append({'path': str(path), 'reason': type(exc).__name__})
    request = inputs.get('response_summary.json', {})
    if not all(isinstance(request.get(k), int) for k in ('started_ns', 'ended_ns')):
        raise ValueError('response_summary.json must contain the original monotonic request boundaries')
    if request['ended_ns'] < request['started_ns']:
        raise ValueError('invalid request clock range')
    for source in ('npu', 'metrics', 'host', 'process', 'disk'):
        path = run_dir/'samples'/f'{source}.jsonl'
        try:
            with path.open(encoding='utf-8') as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict) or row.get('source', source) != source:
                            raise ValueError('source mismatch')
                        if not all(isinstance(row.get(k), int) for k in ('started_ns', 'ended_ns')) or row['ended_ns'] < row['started_ns']:
                            raise ValueError('missing/invalid sample monotonic boundaries')
                        data = row.get('data')
                        if data is not None and not isinstance(data, list if source in ('npu','process') else dict):
                            raise ValueError('invalid sample data shape')
                        row.update(source=source, evidence_ref=f'samples/{source}.jsonl:{line_number}')
                        rows.append(row)
                    except (ValueError, TypeError) as exc:
                        errors.append({'path': str(path), 'line': line_number, 'reason': str(exc)[:100]})
        except OSError as exc:
            missing.append(f'samples/{source}.jsonl')
            errors.append({'path': str(path), 'reason': type(exc).__name__})
    rows.sort(key=lambda r: (r['started_ns'], r['ended_ns']))
    identity = inputs.get('runtime_identity.json', {})
    expected = [p['pid'] for p in identity.get('processes', [])]
    original = inputs.get('single_request_report.json', {})
    # The stored observer epoch flag is reported evidence, not a new container inspection.
    process_stable = process_epoch_stable(rows, expected)
    epoch = original.get('epoch_stable') is True and process_stable
    device = inputs.get('preflight.json', {}).get('parameters', {}).get('disk_device', 'nvme1n1p1')
    result = summarize(rows, request, epoch_stable=epoch, disk_device=device)
    result['audit_input_status'] = 'COMPLETE_SAMPLES_READ' if not missing and not errors else 'PARTIAL_INPUT'
    result['audit_errors'] = errors
    result['missing_input_files'] = missing
    result['epoch_evidence'] = {'stored_observer_flag': original.get('epoch_stable'),
                                'process_starttimes_stable': process_stable,
                                'new_runtime_inspection': 'NOT_RUN'}
    result['input_run_dir'] = str(run_dir)
    result['reported_metric_deltas'] = inputs.get('metric_deltas.json')
    result['audit_metric_comparison'] = {
        name: {'reported': (inputs.get('metric_deltas.json', {}).get(name) or {}).get('delta'),
               'recomputed': result['metric_deltas'][name]['delta']}
        for name in ('prefix_cache_queries_total', 'prefix_cache_hits_total', 'num_preemptions_total')}
    try:
        result['original_capsule'] = (run_dir/'capsule.txt').read_text(encoding='utf-8')
    except OSError:
        result['original_capsule'] = 'NOT_OBSERVED'
    result['selected_samples'] = {}
    result['sampling_statistics'] = {}
    for source in ('npu', 'metrics', 'host', 'process', 'disk'):
        series = [r for r in rows if r['source'] == source]
        groups = {phase: [r for r in series if classify(r, request) == phase]
                  for phase in ('pre', 'during', 'overlap', 'post')}
        selected = ([max(groups['pre'], key=lambda r: r['ended_ns'])] if groups['pre'] else [])
        selected += groups['during'] + groups['overlap']
        selected += ([max(groups['post'], key=lambda r: r['started_ns'])] if groups['post'] else [])
        selected.sort(key=lambda r: r['started_ns'])
        result['selected_samples'][source] = [project_sample(r, request) for r in selected]
        result['sampling_statistics'][source] = {
            'all_rows': len(series), 'by_clock_phase': {k: len(v) for k, v in groups.items()},
            'invalid_or_unobserved_rows': sum(r.get('status') != 'VERIFIED' for r in series),
            'max_collection_seconds': max(((r['ended_ns']-r['started_ns'])/1e9 for r in series), default=None),
            'max_start_interval_seconds': max(((b['started_ns']-a['started_ns'])/1e9 for a,b in zip(series,series[1:])), default=None)}
    return result, identity


def project_sample(row, request):
    """Project complete JSON by field, never slice serialized JSON strings."""
    record = {k: row.get(k) for k in ('evidence_ref', 'utc', 'started_ns', 'ended_ns', 'status', 'error')}
    record['phase'] = classify(row, request)
    source, data = row['source'], row.get('data')
    if source == 'metrics':
        keys = ('num_requests_running','num_requests_waiting','kv_cache_usage_perc',
                'prefix_cache_queries_total','prefix_cache_hits_total','prefix_cache_queries_created','prefix_cache_hits_created',
                'num_preemptions_total')
        record['data'] = {k: (data or {}).get('vllm:'+k) for k in keys}
    elif source == 'npu':
        record['data'] = {r['device_id']: r['hbm_used_mb_reported'] for r in (data or [])}
    elif source == 'process':
        record['data'] = [{'pid': p['pid'], 'role': p.get('role'), 'rss_bytes': p.get('rss_bytes'),
                           'starttime_ticks': p.get('starttime_ticks'),
                           'io_status': p.get('sources', {}).get('io'),
                           'pss_status': p.get('sources', {}).get('pss')}
                          for p in (data or [])]
    elif source == 'disk':
        record['data'] = {k: (data or {}).get(k) for k in ('major','minor','sectors_read','sectors_written','read_bytes','write_bytes')}
    else:
        record['data'] = data
    return record


def write_audit(output, result, identity):
    output = Path(output)
    result = dict(result)
    selected = result.pop('selected_samples')
    table = ['| NPU | pre | strict during采样峰值 | post | post-pre |', '|---|---:|---:|---:|---:|']
    for r in result['hbm']:
        table.append('| ' + ' | '.join(str(r[k]) if r[k] is not None else 'NOT_OBSERVED'
                     for k in ('device_id','before_mb_reported','during_sample_peak_mb_reported','after_mb_reported','delta_mb_reported')) + ' |')
    common = '这260MB的归属仍UNRESOLVED。内存池留存、首次运行工作区/通信/图相关buffer均是INFERRED候选；本轮无对象事件，不能确定是KV、泄漏、权重搬运或重新加载。'
    technical = ['# C1既有实验证据审计', '', f"来源：`{result['input_run_dir']}`。只读取完整已保存JSON/JSONL，未发POST、未访问API或容器、未提权。",
                 f"输入审计：{result['audit_input_status']}；原实验资源状态：{result['status']}。输入完整不等于物理观测完整。", '',
                 '## 请求、环境与epoch', '', '```json', json.dumps({'request':result['request'],'identity':identity,'epoch':result['epoch_evidence']},ensure_ascii=False,indent=2),'```', '',
                 '## HBM时间线', '', *table, '', '单位npu-smi reported MB；不是KV物理字节。峰值仅来自完整落在客户端窗口的采样。', common, '',
                 '## KV、Host、进程、SSD与原报告核对', '', '```json', json.dumps(result,ensure_ascii=False,indent=2),'```', '',
                 'KV gauge回落与整卡HBM驻留属于不同口径；逻辑usage/回收不等于设备allocation归还。Gauge实际产生分母和释放路径还需定制源码核验。',
                 'Host MemAvailable为全机指标；RSS共享页不可直接累加为独占权重/Engram。设备I/O是分区窗口，不能归因某PID/文件或offload。', '',
                 '## 身份与128/32', '', 'VERIFIED的是保存的客户端run/model/attempt/API响应关联；内部Engine ID、Block generation/group、Rank/Layer/buffer/allocation均NOT_OBSERVED。',
                 'argv与metrics的128/32是两个不同观测字段；配置归一化、定制backend、hybrid group等只能作为待查候选，差异UNRESOLVED。禁止blocks×32/128算物理字节。', '',
                 '## 计算、误差与下一步', '', '窗口delta=最后post值-最后pre值；严格phase由monotonic采集起止判定；overlap独立。样本不是硬件瞬时值，不能将npu-smi启动UTC当分配时刻。',
                 'Stored observer epoch flag只代表原观察器判断；本审计又核对进程starttime，没有复查当前服务。可见reset/created变更使delta未知；有限采样无法排除所有reset或旁路流量。',
                 '先追W4A16/Engram/lazy路径，再核实Logger→Scheduler→KVCacheManager→Coordinator/BlockPool，最后设计selected Rank/Layer的allocation映射。当前不需要再发模型请求。', '',
                 '## 样本引用', '', '```json', json.dumps(selected,ensure_ascii=False,indent=2),'```', '']
    beginner = ['# 这一次请求，我们真正知道了什么？', '', f"这是原run `{Path(result['input_run_dir']).name}`的离线审计，不是第二次模型调用。",
                f"请求：{result['request'].get('status')}；POST尝试：{result['request'].get('post_attempts')}；输入/输出Token：{result['request'].get('prompt_tokens')} / {result['request'].get('completion_tokens')}。",
                f"输入证据状态：{result['audit_input_status']}；资源结果：{result['status']}。", '',
                'HBM是每张NPU的高速内存；Host DDR是主机内存；SSD是模型文件所在存储。三个总量不是模型状态的三张精确账本。', '', *table, '',
                common, '', '可以把KV使用统计想成“房间有多少位置正被使用”，HBM想成“整栋楼占的面积”。位置腾出来不要求立刻拆掉楼，所以两个数可以一个归零、一个仍高。这个例子只解释口径，不证明本轮特定内存池实现。', '',
                f"Host可用内存变化（字节）：{result['host']['delta_bytes'].get('MemAvailable')}；磁盘读/写增量（字节）：{result['disk']['read_bytes']['delta']} / {result['disk']['write_bytes']['delta']}。这些是窗口变化，不能说是哪块权重/KV在搬运。",
                'Worker I/O/PSS无法读时就是未知，不是0。缓存命中计数为0也不等于模型没使用KV生成。', '',
                '目前请求身份能到API响应；还不知道服务内部请求对应哪块缓存、哪个Layer/Rank上的实际buffer。128/32差异也还未解释。', '',
                '下一步读源码和已有记录，不再跑模型。完整数据、时间线和未知项见[C1技术审计](C1_EVIDENCE_AUDIT.md)。', '']
    (output/'C1_EVIDENCE_AUDIT.md').write_text('\n'.join(technical),encoding='utf-8')
    (output/'C1_EVIDENCE_AUDIT_BEGINNER.md').write_text('\n'.join(beginner),encoding='utf-8')
    (output/'audit_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Keep the exchange file small without ever truncating a value or JSON record.
    compact_identity = {k:identity.get(k) for k in ('id','image','source_metadata','workspace_commit',
                        'workspace_source_kind','observer_sha256','workspace_files_sha256','runtime_loaded_modules','processes')}
    compact_identity['state'] = {k:identity.get('state',{}).get(k) for k in ('Status','StartedAt','Pid')}
    compact_processes = [{'pid':p['pid'],'role':p['role'],'identity_stable':p['identity_stable'],
                          'rss_before':p['before'].get('rss_bytes'),'rss_after':p['after'].get('rss_bytes'),
                          'pss_before':p['before'].get('pss_bytes'),'pss_after':p['after'].get('pss_bytes'),
                          'io_deltas':p['io_deltas'], 'before_ref':p['before'].get('sources'),
                          'after_ref':p['after'].get('sources')} for p in result['processes']]
    compact_timeline = {k: v for k,v in selected.items() if k!='process'}
    compact_timeline['process'] = [{k:v for k,v in row.items() if k!='data'} |
                                  {'rss_by_pid':{str(p['pid']):p['rss_bytes'] for p in row['data']}}
                                  for row in selected['process']]
    small = {'format':'COMPLETE_RUN_PROJECTED_FIELDS_V1','input_run_dir':result['input_run_dir'],
             'request':result['request'],'identity':compact_identity,'audit_input_status':result['audit_input_status'],
             'errors':result['audit_errors'],'missing':result['missing_input_files'],
             'hbm':result['hbm'],'host':{k:v for k,v in result['host'].items() if k!='during'},
             'processes':compact_processes,'disk':result['disk'],'metrics':result['metric_deltas'],
             'sampling_statistics':result['sampling_statistics'],'epoch_evidence':result['epoch_evidence'],
             'cache_config_info':result['cache_config_info'], 'original_capsule':result['original_capsule'],
             'timeline':compact_timeline}
    encoded = json.dumps(small,ensure_ascii=False,separators=(',',':'))
    if len(encoded.encode('utf-8')) > 19000:
        # Preserve PID/RSS/starttime and permission evidence; skip unrelated optional fields.
        for p in small['processes']:
            for boundary in ('before_ref','after_ref'):
                p[boundary] = {k:v for k,v in (p[boundary] or {}).items() if k in ('io','pss')}
        encoded=json.dumps(small,ensure_ascii=False,separators=(',',':'))
    if len(encoded.encode('utf-8')) > 20000:
        small['timeline_selection']='BOUNDARIES_FIRST_LAST_DURING_AND_OVERLAP; complete selected timeline remains in technical report'
        for source, series in small['timeline'].items():
            during = [r for r in series if r['phase']=='during']
            small['timeline'][source]=[r for r in series if r['phase']!='during' or
                                      r is during[0] or r is during[-1]]
        encoded=json.dumps(small,ensure_ascii=False,separators=(',',':'))
    if len(encoded.encode('utf-8')) > 20480:
        raise ValueError('projected summary exceeds 20KB; reports saved, no truncated TXT emitted')
    (output/'C1_AUDIT_SMALL.txt').write_text(encoded+'\n',encoding='utf-8')
