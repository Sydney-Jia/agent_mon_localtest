"""Streaming validation of the factual allocator order. No alternative policy."""
from collections import OrderedDict


class QueueCheck:
    """One Run/PID/Writer epoch/pool in writer order; submits must not be sampled."""

    def __init__(self):
        self.identity = None
        self.queue = OrderedDict()
        self.capacity = None
        self.null_slot = None
        self.seq = -1
        self.multipart = None
        self.request = None
        self.checked_pops = 0
        self.checked_victims = 0
        self.failure = None

    def consume(self, row):
        if self.failure or row.get('event') not in (
                'IMPORTANCE_QUEUE_CHECKPOINT', 'IMPORTANCE_QUEUE_REQUEST', 'IMPORTANCE_QUEUE_EDIT'):
            return
        try:
            self._consume(row)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            self.failure = dict(reason=str(exc), event=row.get('event'),
                                queue_seq=row.get('queue_seq'), writer_seq=row.get('seq_no'))

    def _consume(self, row):
        event = row['event']
        identity = tuple(row[k] for k in ('run_id', 'pid', 'observer_epoch', 'pool_scope'))
        if not all(identity) or (self.identity is not None and identity != self.identity):
            raise ValueError('QUEUE_IDENTITY_DRIFT')
        self.identity = identity
        if row['queue_seq'] != self.seq + 1:
            raise ValueError('QUEUE_SEQUENCE_GAP')
        if self.seq == -1 and event != 'IMPORTANCE_QUEUE_CHECKPOINT':
            raise ValueError('QUEUE_INITIAL_CHECKPOINT_REQUIRED')
        if event == 'IMPORTANCE_QUEUE_REQUEST':
            if self.multipart or self.request is not None:
                raise ValueError('QUEUE_DANGLING_OPERATION')
            self._edges(row)
            self.request = (row['queue_seq'], row['requested_blocks'])
            self.seq += 1
            return
        key = (event, row['queue_seq'], row.get('action'), row.get('request_seq'),
               row.get('slot_count', row['free']), row.get('capacity'), row.get('null_slot'),
               row['free'], row['head'], row['tail'], row['parts'])
        parts = row['parts']
        if not isinstance(parts, int) or parts < 1:
            raise ValueError('INVALID_PART_COUNT')
        if self.multipart is None:
            self.multipart = [key, 0, []]
        if self.multipart[0] != key or row['part'] != self.multipart[1]:
            raise ValueError('MISSING_OR_CHANGED_QUEUE_PART')
        self.multipart[2].extend(row['ordered_slots'] if event == 'IMPORTANCE_QUEUE_CHECKPOINT' else row['slots'])
        self.multipart[1] += 1
        if self.multipart[1] < parts:
            return
        entries = self.multipart[2]
        self.multipart = None
        if event == 'IMPORTANCE_QUEUE_CHECKPOINT':
            if self.seq != -1 or row['initial_indexed'] or row['initial_watched']:
                raise ValueError('INVALID_INITIAL_CHECKPOINT')
            self.capacity, self.null_slot = row['capacity'], row['null_slot']
            self.queue = OrderedDict.fromkeys(entries)
            if (row['free'] != self.capacity - 1 or len(entries) != len(self.queue)
                    or set(entries) != set(range(self.capacity)) - {self.null_slot}):
                raise ValueError('INCOMPLETE_INITIAL_SLOT_SET')
        else:
            if len(entries) != row['slot_count']:
                raise ValueError('SLOT_COUNT_MISMATCH')
            self._apply(row, entries)
        self._edges(row)
        self.seq += 1

    def _apply(self, row, entries):
        action = row['action']
        ids = [entry[0] for entry in entries]
        if any(len(entry) != 4 or entry[2] != 0 for entry in entries):
            raise ValueError('QUEUE_EDIT_WITH_NONZERO_REFERENCE')
        if any(not (0 <= slot < self.capacity) or slot == self.null_slot for slot in ids):
            raise ValueError('INVALID_QUEUE_SLOT')
        if action in ('ALLOCATE_POP', 'INSUFFICIENT_FREE'):
            if self.request is None or self.request[0] != row['request_seq']:
                raise ValueError('QUEUE_REQUEST_UNBOUND')
            count = self.request[1]
            self.request = None
            if action == 'INSUFFICIENT_FREE':
                if ids or count <= len(self.queue):
                    raise ValueError('REJECTION_WITHOUT_CAPACITY_DEFICIT')
                return
            if count != len(ids):
                raise ValueError('ALLOCATED_COUNT_DIFFERS_FROM_REQUEST')
            for slot in ids:
                expected = next(iter(self.queue), None)
                if slot != expected:
                    raise ValueError(f'VICTIM_MISMATCH expected={expected} actual={slot}')
                self.queue.popitem(last=False)
            self.checked_pops += 1
            self.checked_victims += len(ids)
        elif self.request is not None:
            raise ValueError('MISSING_REQUEST_OUTCOME')
        elif action == 'TOUCH_REMOVE':
            for slot in ids:
                if slot not in self.queue:
                    raise ValueError('TOUCHED_SLOT_NOT_QUEUED')
                del self.queue[slot]
        elif action in ('FREE_APPEND', 'FREE_PREPEND', 'UNPIN_APPEND'):
            for slot in reversed(ids) if action == 'FREE_PREPEND' else ids:
                if slot in self.queue:
                    raise ValueError('DOUBLE_RELEASE')
                self.queue[slot] = None
                if action == 'FREE_PREPEND':
                    self.queue.move_to_end(slot, last=False)
        elif action != 'RESET_INDEX' or ids:
            raise ValueError('UNKNOWN_QUEUE_EDIT')

    def _edges(self, row):
        actual = (len(self.queue), next(iter(self.queue), None), next(reversed(self.queue), None))
        if actual != (row['free'], row['head'], row['tail']):
            raise ValueError('QUEUE_COUNT_OR_EDGES_MISMATCH')

    def summary(self):
        status = ('DIVERGED' if self.failure else 'INCOMPLETE' if (
            self.seq < 0 or self.multipart or self.request is not None) else 'MATCHED_THROUGH_CUT')
        return dict(status=status, identity=self.identity, last_complete_seq=self.seq,
                    allocations=self.checked_pops, selected_slots=self.checked_victims,
                    first_error=self.failure)
