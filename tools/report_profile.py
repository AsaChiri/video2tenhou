"""Summarize a running or finished profile without importing pipeline dependencies.

Usage: uv run python tools/report_profile.py work/profile-run
Optional --stage read.run_read --factor 2 estimates the end-to-end ceiling of
making that complete top-level stage twice as fast, with other work unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _events(path):
    # An ongoing writer can leave an incomplete last line. Never ignore a
    # malformed completed line, which would silently falsify the report.
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            yield json.loads(line)


def _number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def interval_union_seconds(intervals):
    """Measure wall time covered by half-open intervals, counting overlaps once.

    Nested searches and touching endpoints do not add occupied wall time.
    Reversed intervals are invalid; callers must not silently repair telemetry.
    """
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Interval ends before it starts")
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def summarize(directory: Path, stages=(), factor=2.0):
    """Read recorded data only; unfinished spans remain visibly incomplete.

    Timed-out status is not inferred from wall time alone. FEASIBLE/UNKNOWN
    searches are reported separately from near-budget searches, whose elapsed
    time reached 90% of their configured limit. Neither proves why search stopped.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")
    directory = Path(directory)
    summary = _load(directory / "summary.json")
    metadata = _load(directory / "metadata.json")
    events = list(_events(directory / "events.jsonl"))
    main_thread = next(
        (
            row["thread"]
            for row in events
            if row.get("name") == "cli.cmd_convert" and row["event"] == "start"
        ),
        None,
    )
    stage_seconds = defaultdict(float)
    stage_calls = Counter()
    hands = {}
    searches = {}
    search_intervals = defaultdict(list)
    role_intervals = defaultdict(list)
    dense_seconds = defaultdict(float)
    active = {}
    samples = []
    gpu = defaultdict(lambda: defaultdict(list))
    for row in events:
        if row["event"] == "start":
            active[(row["thread"], row["name"])] = row
        elif row["event"] == "end":
            active.pop((row["thread"], row["name"]), None)
            if row.get("name") == row.get("stage"):
                stage_seconds[row["name"]] += row["seconds"]
                stage_calls[row["name"]] += 1
            if row.get("name") == "decode.decode_hand":
                hand = hands.setdefault(
                    str(row["hand"]), dict(seconds=0.0, calls=0, failures=0)
                )
                hand["seconds"] += row["seconds"]
                hand["calls"] += 1
                hand["failures"] += int(row["failed"])
            if row.get("name") == "read.dense_reads":
                dense_seconds[row.get("hand")] += row["seconds"]
        elif row["event"] == "cp_search":
            role = (
                "main_or_diagnostic" if row["thread"] == main_thread else "alternative"
            )
            key = (role, row.get("hand"), row.get("stage"))
            item = searches.setdefault(
                key,
                dict(
                    role=role,
                    hand=row.get("hand"),
                    stage=row.get("stage"),
                    calls=0,
                    seconds=0.0,
                    statuses=Counter(),
                    budgets=Counter(),
                    near_budget=0,
                ),
            )
            item["calls"] += 1
            item["seconds"] += row["seconds"]
            item["statuses"][row["status"]] += 1
            item["budgets"][str(row["time_budget"])] += 1
            item["near_budget"] += int(
                row["time_budget"] > 0 and row["seconds"] >= 0.9 * row["time_budget"]
            )
            # The profiler writes each event immediately after its search.
            # Recover intervals on its common monotonic elapsed-time axis.
            interval = (row["elapsed"] - row["seconds"], row["elapsed"])
            search_intervals[key].append(interval)
            role_intervals[role].append(interval)
        elif row["event"] == "resources":
            samples.append(row)
            for line in row.get("gpu_csv", []):
                fields = next(csv.reader([line], skipinitialspace=True))
                if len(fields) != 7:
                    continue
                key = (fields[0], fields[1], row.get("stage"))
                for name, value in zip(
                    (
                        "utilization_percent",
                        "memory_utilization_percent",
                        "used_memory_mib",
                        "total_memory_mib",
                        "power_watts",
                    ),
                    fields[2:],
                    strict=False,
                ):
                    value = _number(value)
                    if value is not None:
                        gpu[key][name].append(value)
    wall = summary.get("conversion_wall_seconds")
    timings = summary.get("timings", [])
    detailed = [
        row
        for row in timings
        if row["name"] not in stage_seconds and not row["name"].startswith("profile.")
    ]
    for key, item in searches.items():
        item["active_wall_seconds"] = interval_union_seconds(search_intervals[key])
    output = dict(
        finished=bool(summary),
        failure=summary.get("failure"),
        wall_seconds=wall,
        process_cpu_seconds=summary.get("conversion_process_cpu_seconds"),
        stages=[
            dict(
                name=name,
                seconds=seconds,
                calls=stage_calls[name],
                wall_fraction=seconds / wall if wall else None,
            )
            for name, seconds in sorted(
                stage_seconds.items(), key=lambda item: -item[1]
            )
        ],
        per_hand=hands,
        cp_searches=list(searches.values()),
        inclusive_timings=detailed,
        cp_active_wall_seconds_by_role={
            role: interval_union_seconds(intervals)
            for role, intervals in role_intervals.items()
        },
        dense_read_wall_seconds=sum(dense_seconds.values()),
        dense_read_wall_seconds_by_hand={
            str(hand): seconds for hand, seconds in dense_seconds.items()
        },
        active_spans=list(active.values()),
        resources=summary.get("resources", {}),
        cache=metadata.get("cache", {}),
        resource_samples=len(samples),
        gpu=[
            dict(
                index=key[0],
                name=key[1],
                stage=key[2],
                metrics={
                    name: dict(
                        samples=len(values),
                        mean=sum(values) / len(values),
                        maximum=max(values),
                    )
                    for name, values in metrics.items()
                },
            )
            for key, metrics in gpu.items()
        ],
    )
    if wall and output["process_cpu_seconds"] is not None:
        output["occupied_logical_cores"] = output["process_cpu_seconds"] / wall
        cpus = metadata.get("logical_cpus")
        output["machine_cpu_fraction"] = (
            output["occupied_logical_cores"] / cpus if cpus else None
        )
    if stages:
        missing = sorted(set(stages) - set(stage_seconds))
        if missing:
            output["amdahl"] = dict(
                unavailable="Selected stages have not completed", stages=missing
            )
        elif wall:
            fraction = sum(stage_seconds[name] for name in set(stages)) / wall
            if fraction > 1.00001:
                raise ValueError(
                    "Selected stage time exceeds conversion wall; cannot use overlapping times for Amdahl"
                )
            output["amdahl"] = dict(
                stages=sorted(set(stages)),
                stage_fraction=fraction,
                assumed_stage_speedup=factor,
                predicted_total_speedup=1 / (1 - fraction + fraction / factor),
                infinite_stage_speedup_ceiling=None
                if fraction >= 1
                else 1 / (1 - fraction),
            )
        else:
            output["amdahl"] = dict(unavailable="Conversion has not finished")
    return output


def render(report):
    """Produce a compact Markdown explanation with explicit overlap limitations."""
    lines = [
        "# Conversion profile",
        "",
        "Finished."
        if report["finished"]
        else "Still running; only completed spans are shown.",
        "",
    ]
    if report["failure"]:
        lines += [f"Run ended with an error: `{report['failure']}`", ""]
    if report["wall_seconds"] is not None:
        lines += [f"Conversion wall time: **{report['wall_seconds']:.1f} s**.", ""]
    if "occupied_logical_cores" in report:
        lines += [
            f"Main-process CPU averaged {report['occupied_logical_cores']:.2f} logical cores (includes its threads; excludes ffmpeg child CPU).",
            "",
        ]
    lines += ["| Top-level stage | Seconds | Share of conversion |", "|---|---:|---:|"]
    for row in report["stages"]:
        fraction = (
            f"{row['wall_fraction']:.1%}"
            if row["wall_fraction"] is not None
            else "pending"
        )
        lines.append(f"| {row['name']} | {row['seconds']:.1f} | {fraction} |")
    lines += [
        "",
        "## Search",
        "",
        "| Hand | Role | Calls | Active wall seconds | Inclusive seconds | Status counts | Near budget |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in report["cp_searches"]:
        lines.append(
            f"| {row['hand']} | {row['role']} | {row['calls']} | {row['active_wall_seconds']:.1f} | {row['seconds']:.1f} | {dict(row['statuses'])} | {row['near_budget']} |"
        )
    lines += [
        "",
        "Active search wall time merges overlapping intervals on the profiler's monotonic clock. It counts each instant once within a role, including waits during a native solver call; it is not CPU time. Inclusive seconds sum all calls and can exceed wall time. Active calls that have not returned are absent until their completion events arrive.",
        "",
    ]
    for role, seconds in report["cp_active_wall_seconds_by_role"].items():
        lines.append(f"- {role}: {seconds:.1f} s active wall time.")
    lines += [
        "",
        f"Completed dense reads: {report['dense_read_wall_seconds']:.1f} s wall time. These calls are serial on the decoder thread; the total includes cache access and actual video/recognition work.",
        "",
        "Near-budget means at least 90% of the configured wall limit; FEASIBLE/UNKNOWN alone does not prove a timeout. Different roles and nested operations must not be added unless their intervals are disjoint.",
        "",
        "## Nested work",
        "",
        "| Stage | Hand | Operation | Calls | Inclusive seconds |",
        "|---|---|---|---:|---:|",
    ]
    for row in sorted(report["inclusive_timings"], key=lambda item: -item["seconds"]):
        lines.append(
            f"| {row.get('stage')} | {row['hand']} | {row['name']} | {row['calls']} | {row['seconds']:.1f} |"
        )
    lines += [
        "",
        "These nested timings are not additive. Dense-read time includes sampling, recognition and cache access; its nested detector/classifier calls and frame waits show actual work. This profiler does not identify cache hits directly.",
        "",
        "## Per hand",
        "",
        "| Hand | Decode wall seconds | Calls | Failures |",
        "|---|---:|---:|---:|",
    ]
    for hand, row in sorted(report["per_hand"].items(), key=lambda item: int(item[0])):
        lines.append(
            f"| {hand} | {row['seconds']:.1f} | {row['calls']} | {row['failures']} |"
        )
    lines += [
        "",
        f"Resource samples: {report['resource_samples']}. GPU summary in JSON reports device-wide sample means and maxima by stage; unrelated applications are included. Sampled process-tree memory is summed RSS, which can double-count shared pages; short-lived child CPU can be missed.",
        "",
    ]
    if "amdahl" in report:
        lines += [
            "## Conditional optimization estimate",
            "",
            "```json",
            json.dumps(report["amdahl"], indent=2),
            "```",
            "",
            "This assumes unchanged quality, workload and other-stage time. It is a bound based on this measured run, not a promised speedup.",
            "",
        ]
    return "\n".join(lines)


def main(argv=None):
    """Write report.json/report.md beside input events without changing the profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Completed top-level stage for conditional speedup estimate; repeatable",
    )
    parser.add_argument("--factor", type=float, default=2.0)
    args = parser.parse_args(argv)
    report = summarize(args.profile, args.stage, args.factor)
    (args.profile / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.profile / "report.md").write_text(render(report), encoding="utf-8")
    print(args.profile / "report.md")


if __name__ == "__main__":
    main()
