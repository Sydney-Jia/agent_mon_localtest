"""Pinned source contract from KV_SOURCE_20261009-162613-3338923.txt."""

EXPECTED_VERSIONS = {'vllm': '0.27.1+empty', 'vllm-ascend': '0.1.dev5097+ge43cf1e9f'}
KV_FILE_SHA256 = '70f7f608c0963af155540630a5633483e5c19c0c0280dfd621760d5db2a419a6'
KV_MODULE = 'vllm.v1.core.kv_cache_manager'
KV_SIGNATURES = {
    'get_computed_blocks': ('self', 'request'),
    'allocate_slots': ('self', 'request', 'num_new_tokens', 'num_new_computed_tokens',
                       'new_computed_blocks', 'num_lookahead_tokens',
                       'num_external_computed_tokens', 'delay_cache_blocks',
                       'num_encoder_tokens', 'full_sequence_must_fit', 'reserved_blocks',
                       'has_scheduled_reqs'),
    'free': ('self', 'request'),
    'get_blocks': ('self', 'request_id'),
}
TARGETS = {
    'vllm.entrypoints.openai.chat_completion.serving': (
        ('OpenAIServingChat', 'create_chat_completion', 'api_chat'),),
    'vllm.v1.engine.input_processor': (
        ('InputProcessor', 'assign_request_id', 'bind'),),
    'vllm.v1.request': (('Request', 'from_engine_core_request', 'convert'),),
    'vllm.v1.engine.core': (('EngineCore', 'add_request', 'engine_add'),),
    'vllm.v1.core.sched.scheduler': (
        ('Scheduler', 'add_request', 'scheduler_add'),
        ('Scheduler', '_free_request', 'scheduler_free'),
        ('Scheduler', 'finish_requests', 'scheduler_finish'),),
    KV_MODULE: (
        ('KVCacheManager', 'get_computed_blocks', 'lookup'),
        ('KVCacheManager', 'allocate_slots', 'allocate'),
        ('KVCacheManager', 'free', 'free'),),
}
