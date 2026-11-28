# Observe native vLLM KV cache events

The collector supports the native three-frame MessagePack protocol verified with vLLM revision `c4df37` and its per-rank TCP port offsets. It listens on loopback in the same network namespace as the model service. Collection starts only when you explicitly run the script and enable KV events on the service.

Install the optional transport dependencies and start the collector from `kv_ana/` before starting the publisher. This example uses two DP ranks. For one replica use `--ranks 0`.

```bash
python -m pip install '.[kv-events]'
python scripts/collect_vllm_kv_events.py capture \
  --out kv-event-run --seconds 1800 --base-port 5557 --ranks 0 1
```

When both sockets are ready, start the service with these fields in its vLLM YAML. If the ports are already in use, change the endpoint port and `--base-port` together. The service connects to `127.0.0.1:5557` for DP0 and `127.0.0.1:5558` for DP1.

```yaml
kv-events-config:
  enable_kv_cache_events: true
  publisher: zmq
  endpoint: tcp://127.0.0.1:5557
  topic: vllm-kv-events
```

The capture writes `events.jsonl` and `summary.json`. It keeps block hashes, group metadata, rank, sequence numbers, and event times. Prompt token IDs, session IDs, and extra hash keys are omitted. Use a new output path to summarize again without the transport dependencies.

```bash
python scripts/collect_vllm_kv_events.py summarize \
  --events kv-event-run/events.jsonl --out group-summary.json --ranks 0 1
```

The summary tracks `BlockStored` and `BlockRemoved` by rank, group, and storage scope. It reports time between the first visible store and a matching remove when the observations are continuous. Missing or invalid sequence data invalidates the affected tracked entries. `AllBlocksCleared` is counted as a reset. `no_removal_observed` describes only the received events. The subscriber cannot detect loss after the last received sequence. Cache index events alone are insufficient to measure physical KV writes or attribute reuse to individual requests.
