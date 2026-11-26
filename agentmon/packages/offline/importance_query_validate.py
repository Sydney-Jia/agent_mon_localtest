"""Verify real request hash series and inner/outer lookup joins, in Writer order."""
import hashlib
import sqlite3


EVENTS = frozenset({"KV_DEMAND_HASH_CHUNK", "KV_DEMAND_VERSION", "KV_DEMAND_QUERY",
                    "KV_QUERY_GROUP_RESULTS", "HBM_ELIGIBILITY_BEGIN",
                    "HBM_ELIGIBILITY_PROBE", "HBM_ELIGIBILITY_END"})


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


class QueryCheck:
    """One Run/PID/epoch. The native hash vector lives in a temporary SQLite file."""

    def __init__(self):
        self.db = sqlite3.connect("")
        self.db.execute("CREATE TABLE hashes(s TEXT, idx INTEGER, raw BLOB, PRIMARY KEY(s,idx)) WITHOUT ROWID")
        self.identity = None
        self.seq = -1
        self.series = {}
        self.active = {}
        self.failure = None
        self.versions = self.ended = self.probes = self.misses = self.skipped = self.unknown_leases = 0
        self.admissions = self.streaming_updates = self.query_only = self.short_vectors = 0

    def close(self):
        self.db.close()

    def consume(self, row):
        if self.failure or row.get("event") not in EVENTS:
            return
        try:
            writer = (row["run_id"], row["pid"], row["observer_epoch"])
            require(all(v is not None and v != "" for v in writer), "MISSING_WRITER_IDENTITY")
            if self.identity is None:
                self.identity = writer
            require(writer == self.identity and type(row["seq_no"]) is int and row["seq_no"] > self.seq,
                    "WRITER_SCOPE_OR_ORDER")
            self.seq = row["seq_no"]
            require(isinstance(row["request_id"], str) and row["request_id"], "MISSING_REQUEST_ID")
            event = row["event"]
            if event.startswith("KV_DEMAND_") and event != "KV_DEMAND_QUERY":
                self._series(row)
            else:
                self._query(row)
        except (KeyError, ValueError, IndexError, TypeError, OverflowError, sqlite3.IntegrityError) as exc:
            self.failure = dict(reason=str(exc), event=row["event"], seq_no=row.get("seq_no"),
                                request_id=row.get("request_id"), query_id=row.get("query_id"))

    def _series(self, row):
        sid = row["series_id"]
        owner = (row["request_id"], row["pool_scope"], row["hash_block_tokens"])
        require(sid and row["pool_scope"] and type(owner[2]) is int and owner[2] > 0, "DEMAND_DOMAIN")
        series = self.series.setdefault(sid, dict(owner=owner, count=0, committed=None, chain=bytes(32)))
        require(series["owner"] == owner, "SERIES_DOMAIN_CHANGED")
        count = series["count"]
        if row["event"] == "KV_DEMAND_HASH_CHUNK":
            enc = row["base_hashes_hex"]
            require(isinstance(enc,list) and 1 <= len(enc) <= 128, "CHUNK_FORMAT")
            require(row["index_start"] == count and row["previous_chain_sha256"] == series["chain"].hex(),
                    "MISSING_OR_DUPLICATE_HASH_CHUNK")
            chain = series["chain"]
            for offset, value in enumerate(enc):
                raw = bytes.fromhex(value)
                require(raw and raw.hex() == value, "HASH_ENCODING")
                self.db.execute("INSERT INTO hashes VALUES(?,?,?)", (sid,count+offset,raw))
                chain = hashlib.sha256(chain + len(raw).to_bytes(4,"big") + raw).digest()
            require(chain.hex() == row["chain_sha256"], "CHUNK_DIGEST")
            series.update(chain=chain,count=count+len(enc))
        else:
            require(row["parent_count"] == (series["committed"] or 0) and row["hash_count"] == count
                    and row["chain_sha256"] == series["chain"].hex()
                    and row["hash_covered_tokens"] == count * owner[2], "DEMAND_VERSION_COVERAGE")
            require(count != series["committed"], "DUPLICATE_DEMAND_VERSION")
            self.admissions += int(row["phase"] == "ADMISSION")
            self.streaming_updates += int(row["phase"] == "STREAMING_UPDATE")
            self.query_only += int(row["phase"] == "QUERY" and series["committed"] is None)
            # An intentionally un-hashed request still has real input; omitted
            # positions must not silently enter an alternative policy as MISS.
            self.short_vectors += int(count < row["all_token_count"] // owner[2])
            series["committed"] = count
            self.versions += 1

    def _query(self, row):
        event, qid = row["event"], row["query_id"]
        if event == "HBM_ELIGIBILITY_BEGIN":
            require(qid not in self.active and bool(qid), "DUPLICATE_QUERY_BEGIN")
            self.active[qid] = dict(request=row["request_id"], groups=row["group_ids"],
                                    max_length=row["max_length"], probes=0, pending=None, sid=None)
            return
        query = self.active.get(qid)
        require(query and query["request"] == row["request_id"], "QUERY_WITHOUT_BEGIN_OR_WRONG_REQUEST")
        if event == "KV_DEMAND_QUERY":
            sid = row["series_id"]
            series = self.series.get(sid)
            require(query["sid"] is None and series and series["owner"][:2] ==
                    (row["request_id"],row["pool_scope"]) and series["committed"] == row["hash_count"] ==
                    series["count"] and row["chain_sha256"] == series["chain"].hex(), "QUERY_HASH_VECTOR_UNPROVEN")
            require(row["group_ids"] == query["groups"] and row["max_length"] == query["max_length"],
                    "QUERY_RULE_MISMATCH")
            query.update(sid=sid, count=row["hash_count"])
            return
        require(query["sid"] is not None, "MISSING_DEMAND_QUERY_BINDING")
        series = self.series[query["sid"]]
        if event == "KV_QUERY_GROUP_RESULTS":
            require(row["pool_scope"] == series["owner"][1] and query["pending"] is None
                    and row["ordinal"] == query["probes"]+1, "INNER_LOOKUP_BINDING")
            query["pending"] = row
        elif event == "HBM_ELIGIBILITY_PROBE":
            inner = query["pending"]
            require(inner and inner["ordinal"] == row["ordinal"], "MISSING_INNER_LOOKUP")
            end, size = row["token_end"], series["owner"][2]
            require(type(end) is int and end > row["token_start"] >= 0 and end % size == 0
                    and end // size <= query["count"], "LOOKUP_OUTSIDE_DEMAND")
            raw = self.db.execute("SELECT raw FROM hashes WHERE s=? AND idx=?",
                                  (query["sid"],end//size-1)).fetchone()[0]
            expected = [(group,hashlib.sha256(raw+group.to_bytes(4,"big")).hexdigest()) for group in query["groups"]]
            require([(k["group_id"],k["key_digest"]) for k in row["keys"]] == expected, "LOOKUP_KEY_DIFFERS_FROM_DEMAND")
            visited = inner["visited"]
            require([(k["group_id"],k["key_digest"]) for k in visited] == expected[:len(visited)]
                    and inner["unvisited_group_ids"] == query["groups"][len(visited):], "GROUP_ORDER_OR_KEY")
            all_hit = row["hit"] is True
            require(type(row["hit"]) is bool and visited and len(visited) <= len(expected), "JOINT_RESULT_FORMAT")
            hits = visited if all_hit else visited[:-1]
            require(all(k["result"] == "HIT" and k["block_id"] is not None for k in hits), "INNER_HIT_FORMAT")
            for hit in hits:
                lease = hit["allocation_lease"]
                require((lease is None and hit["lease_status"] == "UNKNOWN") or
                        (type(lease) is int and lease > 0 and hit["lease_status"] == "OBSERVED"),
                        "INNER_HIT_LEASE")
                self.unknown_leases += int(lease is None)
            if all_hit:
                require(inner["termination"] == "ALL_VISITED" and inner["first_missing_group"] is None and
                        not inner["unvisited_group_ids"] and [k["block_id"] for k in hits] ==
                        row["returned_block_ids"], "INNER_OUTER_HIT_MISMATCH")
            else:
                miss = visited[-1]
                require(inner["termination"] == "FIRST_MISS" and miss["result"] == "MISS" and
                        miss["group_id"] == inner["first_missing_group"] and miss["block_id"] is None and
                        miss["allocation_lease"] is None and miss["lease_status"] == "NOT_APPLICABLE" and
                        row["returned_block_ids"] == [], "INNER_OUTER_MISS_MISMATCH")
                self.misses += 1
            self.skipped += len(inner["unvisited_group_ids"])
            self.probes += 1
            query.update(pending=None, probes=query["probes"]+1)
        else:
            require(query["pending"] is None and row["probes"] == query["probes"], "QUERY_END_COUNT")
            self.ended += 1
            del self.active[qid]

    def summary(self):
        incomplete = sum(v["committed"] != v["count"] for v in self.series.values())
        status = ("DIVERGED" if self.failure else "INCOMPLETE" if
                  (not self.versions or not self.ended or not self.probes or self.active or incomplete
                   or self.unknown_leases or self.short_vectors)
                  else "MATCHED_THROUGH_CUT")
        return dict(status=status, writer=self.identity, hash_versions=self.versions, closed_queries=self.ended,
                    verified_probes=self.probes, joint_misses=self.misses, unvisited_groups=self.skipped,
                    unknown_hit_leases=self.unknown_leases, open_queries=len(self.active),
                    admissions=self.admissions, streaming_updates=self.streaming_updates,
                    series_first_seen_at_query=self.query_only, incomplete_raw_hash_vectors=self.short_vectors,
                    incomplete_series=incomplete, first_error=self.failure)
