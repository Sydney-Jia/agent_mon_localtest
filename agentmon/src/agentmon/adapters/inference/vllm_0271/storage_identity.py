"""Shared, process-local storage identity; never an address/content identity.

Inventory and execution views must call the SAME registry. A storage handle is
used only internally, and retained Tensor references are weak. Losing all live
witnesses breaks continuity deliberately, even if an allocator reuses a handle.
"""
import os
import threading
import uuid
import weakref

CONTRACT = 'PROCESS_LIVE_STORAGE_REGISTRY_V1'


class StorageIdentityRegistry:
    def __init__(self):
        self._reset()

    def _reset(self):
        self.pid = os.getpid()
        self.registry_id = uuid.uuid4().hex
        self.entries = {}
        self.lock = threading.RLock()
        self.calls = 0

    @staticmethod
    def _key(tensor):
        storage = tensor.untyped_storage()
        handle = getattr(storage, '_cdata', None)
        if type(handle) is not int:
            raise ValueError('STORAGE_IMPLEMENTATION_IDENTITY_NOT_OBSERVED')
        # A changed extent/device creates a new observation generation. Neither
        # this generation nor the opaque UUID denotes KV content generation.
        return handle, str(tensor.device), int(storage.nbytes())

    def _alive(self, key, refs):
        alive = []
        for ref in refs:
            tensor = ref()
            try:
                if tensor is not None and self._key(tensor) == key:
                    alive.append(ref)
            except Exception:
                pass  # an unavailable witness cannot establish continuity
        return alive

    def describe(self, tensor):
        if self.pid != os.getpid():
            self._reset()  # do not reuse a parent's UUID or inherited lock
        with self.lock:
            key = self._key(tensor)
            old = self.entries.get(key)
            # A single live witness establishes continuity. Avoid re-reading
            # every alias's device/storage on every layer of every decode step.
            refs = []
            if old:
                for ref in old[1]:
                    if self._alive(key, [ref]):
                        refs = old[1]
                        break
            identity = old[0] if refs else uuid.uuid4().hex
            if not any(r() is tensor for r in refs):
                refs.append(weakref.ref(tensor))
            self.entries[key] = identity, refs
            self.calls += 1
            if self.calls % 1024 == 0:
                self.entries = {k: (v[0], live) for k, v in self.entries.items()
                                if (live := self._alive(k, v[1]))}
            return dict(storage_object_id=identity,
                        storage_registry_id=self.registry_id,
                        storage_identity_scope=CONTRACT,
                        storage_identity_status='VERIFIED')
