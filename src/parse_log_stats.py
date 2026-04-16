"""Parse vllm-omni --log-stats output for Ming serving benchmarks.

The parser consumes the actual PrettyTable-style logs emitted by
vllm_omni.metrics.stats and summarizes per-request E2E, per-stage timings,
and per-edge transfer timings into a compact JSON report.
"""

import argparse
import json
import re
from pathlib import Path


TITLE_RE = re.compile(
    r"\[(?P<title>Overall Summary|RequestE2EStats \[request_id=(?P<e2e_id>[^\]]+)\]|"
    r"StageRequestStats \[request_id=(?P<stage_id>[^\]]+)\]|"
    r"TransferEdgeStats \[request_id=(?P<transfer_id>[^\]]+)\])\]"
)


def _coerce_value(text: str):
    value = text.strip()
    if value == "":
        return value
    normalized = value.replace(",", "")
    try:
        if any(ch in normalized for ch in (".", "e", "E")):
            return float(normalized)
        return int(normalized)
    except ValueError:
        return value


def _parse_prettytable(lines: list[str]) -> list[dict]:
    rows = []
    for line in lines:
        marker_positions = [pos for pos in (line.find("|"), line.find("+")) if pos >= 0]
        stripped = line[min(marker_positions) :].strip() if marker_positions else line.strip()
        if not stripped.startswith("|"):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        rows.append(parts)

    if len(rows) < 2:
        return []

    headers = rows[0]
    data_rows = []
    for row in rows[1:]:
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        data_rows.append(dict(zip(headers, [_coerce_value(v) for v in row], strict=False)))
    return data_rows


def _collect_table_block(lines: list[str], start_idx: int) -> tuple[list[dict], int]:
    table_lines = []
    idx = start_idx + 1
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped:
            idx += 1
            continue
        if TITLE_RE.search(lines[idx]):
            break
        if "|" in stripped or "+" in stripped:
            marker_positions = [pos for pos in (lines[idx].find("|"), lines[idx].find("+")) if pos >= 0]
            table_lines.append(lines[idx][min(marker_positions) :].strip() if marker_positions else stripped)
            idx += 1
            continue
        if table_lines:
            break
        idx += 1
    return _parse_prettytable(table_lines), idx


def parse_log_file(path: str) -> dict:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    parsed = {
        "overall_summary": None,
        "request_e2e": [],
        "stage_request": [],
        "transfer_edge": [],
    }

    idx = 0
    while idx < len(lines):
        match = TITLE_RE.search(lines[idx])
        if not match:
            idx += 1
            continue

        table_rows, next_idx = _collect_table_block(lines, idx)
        title = match.group("title")

        if title == "Overall Summary":
            parsed["overall_summary"] = table_rows
        elif match.group("e2e_id") is not None:
            parsed["request_e2e"].append({"request_id": match.group("e2e_id"), "rows": table_rows})
        elif match.group("stage_id") is not None:
            parsed["stage_request"].append({"request_id": match.group("stage_id"), "rows": table_rows})
        elif match.group("transfer_id") is not None:
            parsed["transfer_edge"].append({"request_id": match.group("transfer_id"), "rows": table_rows})

        idx = next_idx

    return parsed


def _single_row_table_to_dict(rows: list[dict]) -> dict:
    result = {}
    for row in rows:
        field = row.get("Field")
        if field in (None, ""):
            continue
        result[str(field)] = row.get("Value")
    return result


def _multi_column_table_to_rows(rows: list[dict], column_name: str) -> list[dict]:
    expanded = []
    if not rows:
        return expanded

    column_keys = [key for key in rows[0] if key != "Field"]
    values_by_column = {key: {} for key in column_keys}
    for row in rows:
        field = row.get("Field")
        if field in (None, ""):
            continue
        for column in column_keys:
            values_by_column[column][str(field)] = row.get(column)

    for column in column_keys:
        expanded.append({column_name: column, **values_by_column[column]})
    return expanded


def _numeric_mean(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return float(sum(values) / len(values))


def _build_summary(parsed: dict) -> dict:
    overall_rows = parsed["overall_summary"] or []
    overall = _single_row_table_to_dict(overall_rows)

    e2e_rows = []
    for item in parsed["request_e2e"]:
        rows = _single_row_table_to_dict(item["rows"])
        rows["request_id"] = item["request_id"]
        e2e_rows.append(rows)

    stage_rows = []
    for item in parsed["stage_request"]:
        for row in _multi_column_table_to_rows(item["rows"], "stage_id"):
            row["request_id"] = item["request_id"]
            stage_rows.append(row)

    transfer_rows = []
    for item in parsed["transfer_edge"]:
        for row in _multi_column_table_to_rows(item["rows"], "edge"):
            row["request_id"] = item["request_id"]
            tx = row.get("tx_time_ms")
            rx = row.get("rx_decode_time_ms")
            flight = row.get("in_flight_time_ms")
            size_kbytes = row.get("size_kbytes")
            if isinstance(tx, (int, float)) and isinstance(rx, (int, float)) and isinstance(flight, (int, float)):
                row["total_transfer_time_ms"] = float(tx + rx + flight)
            if (
                isinstance(size_kbytes, (int, float))
                and isinstance(row.get("total_transfer_time_ms"), (int, float))
                and row["total_transfer_time_ms"] > 0
            ):
                row["throughput_mbps"] = float((size_kbytes * 8.0 / 1024.0) / (row["total_transfer_time_ms"] / 1000.0))
            transfer_rows.append(row)

    per_stage = {}
    for row in stage_rows:
        stage_id = str(row.get("stage_id"))
        per_stage.setdefault(stage_id, []).append(row)

    per_edge = {}
    for row in transfer_rows:
        edge = str(row.get("edge"))
        per_edge.setdefault(edge, []).append(row)

    return {
        "overall_summary": overall,
        "request_count": len(e2e_rows),
        "mean_e2e_total_ms": _numeric_mean(e2e_rows, "e2e_total_ms"),
        "mean_transfers_total_time_ms": _numeric_mean(e2e_rows, "transfers_total_time_ms"),
        "mean_transfers_total_kbytes": _numeric_mean(e2e_rows, "transfers_total_kbytes"),
        "per_stage": {
            stage_id: {
                "mean_stage_gen_time_ms": _numeric_mean(rows, "stage_gen_time_ms"),
                "mean_postprocess_time_ms": _numeric_mean(rows, "postprocess_time_ms"),
                "mean_num_tokens_in": _numeric_mean(rows, "num_tokens_in"),
                "mean_num_tokens_out": _numeric_mean(rows, "num_tokens_out"),
                "mean_audio_generated_frames": _numeric_mean(rows, "audio_generated_frames"),
            }
            for stage_id, rows in per_stage.items()
        },
        "per_edge": {
            edge: {
                "mean_size_kbytes": _numeric_mean(rows, "size_kbytes"),
                "mean_tx_time_ms": _numeric_mean(rows, "tx_time_ms"),
                "mean_rx_decode_time_ms": _numeric_mean(rows, "rx_decode_time_ms"),
                "mean_in_flight_time_ms": _numeric_mean(rows, "in_flight_time_ms"),
                "mean_total_transfer_time_ms": _numeric_mean(rows, "total_transfer_time_ms"),
                "mean_throughput_mbps": _numeric_mean(rows, "throughput_mbps"),
            }
            for edge, rows in per_edge.items()
        },
        "raw": {
            "request_e2e": e2e_rows,
            "stage_request": stage_rows,
            "transfer_edge": transfer_rows,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Parse vllm-omni --log-stats output")
    parser.add_argument("log_file", help="Path to the captured server log")
    parser.add_argument("--output-json", default=None, help="Optional path for the parsed JSON summary")
    return parser.parse_args()


def main():
    args = parse_args()
    parsed = parse_log_file(args.log_file)
    summary = _build_summary(parsed)
    output = json.dumps(summary, indent=2, ensure_ascii=False)

    if args.output_json:
        Path(args.output_json).write_text(output, encoding="utf-8")
        print(f"Wrote parsed summary to {args.output_json}")
    else:
        print(output)


if __name__ == "__main__":
    main()
