"""Opt-in fingerprints of the actual Engine request input, never token payloads.

此处不重新分词，也不将消息独立分词后拼接。位置来自转换成功后的
Request.prompt_token_ids；分段只是取证粒度，绝不是 KV Group 的 Block。
密钥仅在内存中使用，不写入事件。指纹用于候选比较，不证明 KV 等价。
"""
import hashlib
import hmac
import os
from pathlib import Path
import struct
import uuid

FLAG = 'AGENTMON_VLLM_0271_INPUT_RESEARCH_ENABLE'
KEY_FILE = 'AGENTMON_VLLM_0271_INPUT_KEY_FILE'
CHUNK_TOKENS = 128
MAX_TOKENS = 262144
ROWS_PER_EVENT = 64


class InputObserver:
    def __init__(self):
        self.enabled = os.environ.get(FLAG, '').lower() in {'1', 'true', 'yes', 'on'}
        self.key = None
        self.problem = None
        if self.enabled:
            try:
                # 有界读取，密钥文件不是配置/Prompt 文件，不将内容或路径写入日志。
                with Path(os.environ[KEY_FILE]).open('rb') as stream:
                    key = stream.read(33)
                if len(key) != 32:
                    raise ValueError('key must be exactly 32 random bytes')
                self.key = key
            except (OSError, KeyError, ValueError):
                self.problem = 'RESEARCH_KEY_UNAVAILABLE_OR_INVALID'

    def observe(self, request, emit):
        """emit(event, **fields); outer Hook fail-open protects model behavior."""
        if not self.enabled:
            return
        if self.problem:
            emit('ENGINE_INPUT_NOT_OBSERVED', reason=self.problem)
            return
        tokens = getattr(request, 'prompt_token_ids', None)
        # 不调用 Tensor/iterator，不读取 embedding。混合输入不能声称完整 token 证据。
        if (getattr(request, 'prompt_embeds', None) is not None
                or getattr(request, 'mm_features', None)
                or getattr(request, 'prompt_is_token_ids', None) is not None):
            emit('ENGINE_INPUT_NOT_OBSERVED', reason='MIXED_OR_MULTIMODAL_INPUT_UNSUPPORTED')
            return
        if type(tokens) is not list or len(tokens) > MAX_TOKENS:
            emit('ENGINE_INPUT_NOT_OBSERVED', reason='TOKEN_LIST_MISSING_OR_LIMIT_EXCEEDED')
            return
        if any(type(t) is not int or t < 0 or t > 0xffffffff for t in tokens):
            emit('ENGINE_INPUT_NOT_OBSERVED', reason='TOKEN_LIST_SCHEMA_UNSUPPORTED')
            return
        capture = uuid.uuid4().hex
        key_id = hmac.new(self.key, b'agentmon-input-key-id-v1', hashlib.sha256).hexdigest()
        common = dict(capture_id=capture, fingerprint_schema='ENGINE_INPUT_HMAC_V1',
                      key_id=key_id, token_count=len(tokens), chunk_tokens=CHUNK_TOKENS)
        emit('ENGINE_INPUT_BEGIN', **common,
             input_basis='SUCCESSFUL_REQUEST_CONVERSION_PROMPT_TOKEN_IDS',
             model_tokenizer_template_identity='NOT_OBSERVED',
             message_source_mapping='NOT_OBSERVED',
             content_validity='NOT_PROVEN_BY_TOKEN_FINGERPRINT',
             raw_tokens_recorded=False)
        prefix = hmac.new(self.key, b'engine-input-prefix-v1\0', hashlib.sha256)
        rows = []
        part = 0
        for start in range(0, len(tokens), CHUNK_TOKENS):
            end = min(start + CHUNK_TOKENS, len(tokens))
            payload = b''.join(struct.pack('>I', t) for t in tokens[start:end])
            prefix.update(payload)
            rows.append(dict(start=start, end=end,
                segment_hmac=hmac.new(self.key, b'engine-input-segment-v1\0' + payload,
                                      hashlib.sha256).hexdigest(),
                prefix_hmac=prefix.copy().hexdigest()))
            if len(rows) == ROWS_PER_EVENT or end == len(tokens):
                emit('ENGINE_INPUT_SEGMENTS', **common, part_index=part, segments=rows)
                rows = []
                part += 1
        emit('ENGINE_INPUT_END', **common, parts=part, input_hmac=prefix.hexdigest(),
             source_mapping='NOT_OBSERVED', kv_equivalence='NOT_PROVEN')
