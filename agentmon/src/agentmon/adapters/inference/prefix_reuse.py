"""Request-side options for reusing vLLM prefixes across Agent turns."""

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any, Mapping


DP_RANK_HEADER = "X-data-parallel-rank"


def affinity_headers(
    headers: Mapping[str, str], session_id: str | None, dp_size: int
) -> dict[str, str]:
    result = dict(headers)
    if not session_id or not session_id.strip() or session_id == "unattached" or dp_size < 2:
        return result
    if any(name.lower() == DP_RANK_HEADER.lower() for name in result):
        return result
    digest = sha256(session_id.encode("utf-8")).digest()
    result[DP_RANK_HEADER] = str(int.from_bytes(digest, "big") % dp_size)
    return result


def stabilize_tool_parameters(
    payload: dict[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep a session's original parameter key order for an equal tool catalog."""
    tools = payload.get("tools")
    previous_tools = reference.get("tools")
    if not isinstance(tools, list) or not isinstance(previous_tools, list):
        return payload
    if "model" in reference and reference["model"] != payload.get("model"):
        return payload
    try:
        # Preserve list order and JSON types, including false versus 0.
        if json.dumps(tools, sort_keys=True, allow_nan=False) != json.dumps(
            previous_tools, sort_keys=True, allow_nan=False
        ):
            return payload
    except (TypeError, ValueError):
        return payload

    updated_tools = None
    for index, (tool, previous_tool) in enumerate(zip(tools, previous_tools, strict=True)):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("parameters"), dict):
            continue
        previous_parameters = previous_tool["function"]["parameters"]
        if json.dumps(function["parameters"]) == json.dumps(previous_parameters):
            continue
        if updated_tools is None:
            updated_tools = tools.copy()
        updated_tools[index] = {
            **tool,
            "function": {**function, "parameters": deepcopy(previous_parameters)},
        }
    return payload if updated_tools is None else {**payload, "tools": updated_tools}
