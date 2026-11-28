import json

import pytest

from kvreuse_b5.observability.vllm_kv_events import decode_frame, summarize, write_summary


TOPIC = b"unit-test"
BLOCK = b"synthetic-key"


def event(kind="BlockStored", group=0, medium="GPU", **fields):
    return {"type": kind, "group_idx": group, "medium": medium, "block_hashes": [BLOCK], **fields}


def batch(rank, sequence, timestamp, *events):
    frames = [TOPIC, sequence.to_bytes(8, "big"), b"fixture"]
    return decode_frame(frames, rank, TOPIC, lambda _: [timestamp, list(events), rank])


def test_projection_omits_request_data():
    record = batch(0, 0, 1.5, event(
        block_hashes=[BLOCK, 2**63], token_ids=[111222333], extra_keys=[["private"]],
        session_id="private", parent_block_hash=b"parent", kv_cache_spec_kind="SlidingWindowSpec",
        kv_cache_spec_sliding_window=128,
    ))
    assert record == {"rank": 0, "seq": 0, "ts": 1.5, "events": [{
        "type": "BlockStored", "group_idx": 0, "medium": "GPU",
        "hashes": ["hex:" + BLOCK.hex(), "int:9223372036854775808"],
        "kv_cache_spec_kind": "SlidingWindowSpec", "kv_cache_spec_sliding_window": 128,
    }]}


def test_messagepack_protocol():
    msgspec = pytest.importorskip("msgspec")
    frames = [TOPIC, (7).to_bytes(8, "big"), msgspec.msgpack.encode([3.25, [event()], 1])]
    record = decode_frame(frames, 1, TOPIC, msgspec.msgpack.decode)
    assert record == batch(1, 7, 3.25, event())


@pytest.mark.parametrize("rank, topic, sequence", [
    (1, TOPIC, b"\0" * 8), (0, b"different", b"\0" * 8), (0, TOPIC, b"\0"),
])
def test_rejects_wrong_rank_topic_or_sequence_width(rank, topic, sequence):
    with pytest.raises(ValueError):
        decode_frame([TOPIC, sequence, b"fixture"], rank, topic, lambda _: [1.0, [], 0])


def test_scope_isolation_and_continuous_age(tmp_path):
    records = [batch(0, 0, 1.5, event(), event(group=1), event(medium="CPU")),
               batch(1, 0, 2.0, event()), batch(0, 1, 2.5, event()),
               batch(0, 2, 5.0, event("BlockRemoved"))]
    capture = tmp_path / "events.jsonl"
    capture.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    output = tmp_path / "summary.json"
    write_summary(capture, output, [0, 1])
    ranks = json.loads(output.read_text(encoding="utf-8"))["ranks"]
    groups = {(row["group_idx"], row["medium"]): row for row in ranks["0"]["groups"]}
    removed = groups[0, "GPU"]
    assert removed["repeated_store_reports"] == 1
    assert removed["tracked_at_last_observation"] == 0
    assert removed["matched_store_to_remove"] == {"count": 1, "min_s": 3.5, "median_s": 3.5, "max_s": 3.5}
    for retained in (groups[1, "GPU"], groups[0, "CPU"], ranks["1"]["groups"][0]):
        assert retained["tracked_at_last_observation"] == 1
        assert retained["matched_store_to_remove"]["count"] == 0
    assert ranks["0"]["removal_evidence"] == "removal_or_reset_observed"
    assert ranks["1"]["removal_evidence"] == "no_removal_observed"


def test_sequence_gaps_prevent_claiming_continuous_residence():
    records = [batch(0, 2, 1.0, event()), batch(0, 4, 3.0, event("BlockRemoved")),
               batch(0, 4, 4.0, event())]
    rank = summarize(records, [0])["ranks"]["0"]
    group = rank["groups"][0]
    assert rank["coverage"] == "incomplete_sequence"
    assert rank["missing_seq_numbers"] == 3
    assert rank["nonincreasing_seq_frames"] == 1
    assert group["invalidated_by_gaps"] == 1
    assert group["unmatched_removals"] == 1
    assert group["matched_store_to_remove"]["count"] == 0


def test_no_batches_and_malformed_capture_are_indeterminate():
    records = [batch(0, 0, 1.0, event()), {"rank": 0, "malformed": True}]
    ranks = summarize(records, [0, 1])["ranks"]
    assert ranks["0"]["malformed_frames"] == 1
    assert ranks["0"]["groups"][0]["tracked_at_last_observation"] == 0
    assert ranks["1"]["coverage"] == "no_batches_received"
    assert all(rank["removal_evidence"] == "indeterminate_due_to_missing_evidence" for rank in ranks.values())


def test_reset_closes_tracking_without_a_removal_time():
    records = [batch(0, 0, 1.0, event(group=None)), batch(0, 1, 2.0, {"type": "AllBlocksCleared"})]
    rank = summarize(records, [0])["ranks"]["0"]
    group = rank["groups"][0]
    assert group["group_idx"] is None
    assert rank["reset_events"] == 1
    assert group["removed_hash_reports"] == 0
    assert group["ended_by_reset"] == 1
    assert group["matched_store_to_remove"]["count"] == 0
