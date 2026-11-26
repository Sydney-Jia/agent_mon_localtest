"""Read KV spec structure, including vLLM's observed scheduler projection.

No vLLM import, tensor access or configuration changes. UniformType is a
container: its summed page size is different from its first member's page size.
The pinned upstream generate_scheduler_kv_cache_config uses that first member.
"""
UNIFORM_CLASS = 'vllm.v1.kv_cache_interface.UniformTypeKVCacheSpecs'
PROJECTION = 'VLLM_0271_UNIFORM_FIRST_MEMBER'
FIELDS = ('block_size', 'num_kv_heads', 'head_size', 'dtype',
          'page_size_bytes', 'sliding_window', 'compression_ratio')


def describe_spec(spec):
    fields = {}
    for name in FIELDS:
        try:
            value = getattr(spec, name, None)
            if isinstance(value, (int, float, bool, str)) or name == 'dtype' and value is not None:
                fields[name] = str(value) if name == 'dtype' else value
        except Exception:
            fields[name] = None
    return {'spec_class': type(spec).__module__+'.'+type(spec).__qualname__, 'spec': fields}


def describe_group_spec(spec, layers):
    raw = describe_spec(spec)
    result = {**raw, 'per_layer_specs': [], 'scheduler_spec_projection': None}
    if raw['spec_class'] != UNIFORM_CLASS:
        return result
    members = getattr(spec, 'kv_cache_specs', None)
    if not isinstance(members, dict) or not members:
        return result
    if not isinstance(layers, list) or set(members) != set(layers) or len(layers) != len(set(layers)):
        return result
    # Preserve actual dict order. Sorting would select a different member than
    # the real scheduler. Capture every member; never infer it from a sum/mean.
    for layer, member in members.items():
        result['per_layer_specs'].append({'layer': layer, **describe_spec(member)})
    first = result['per_layer_specs'][0]
    result['scheduler_spec_projection'] = {
        'contract': PROJECTION, 'status': 'VERIFIED', 'selected_layer': first['layer'],
        'spec_class': first['spec_class'], 'spec': dict(first['spec']),
        'basis': 'OBSERVED_ORDERED_MEMBERS; PINNED_UPSTREAM_FIRST_MEMBER_RULE',
        'upstream_revision': '6e448d0ea9bf3d88d898b65449ca6dc2aec170ac'}
    return result
