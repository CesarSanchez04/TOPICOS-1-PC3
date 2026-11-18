"""Orquestación de los experimentos (la usa el cuaderno). Guarda TODO en results/."""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from baseline import (DENSE_MODEL_ID, GENERATOR_MODEL_ID, RERANKER_MODEL_ID, MAX_NEW_TOKENS, REPETITION_PENALTY,
                      DATA_DIR, PROMPTS_DIR, RESULTS_DIR, read_jsonl, sha256_file)
from controller import Limits, run_task
from evaluation import eval_exp1, eval_memory, summarize_exp1, summarize_memory
from memory import make_memory
from tools import EVIDENCE_SCORE_THRESHOLD


def _jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows))


def run_config(limits: Limits, extra: dict | None = None) -> dict:
    return {
        "executed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(), "python": platform.python_version(),
        "packages": {p: metadata.version(p) for p in ["torch", "transformers", "sentence-transformers",
                                                       "faiss-cpu", "rank-bm25", "pydantic"]},
        "generator": GENERATOR_MODEL_ID, "decoding": {"do_sample": False, "repetition_penalty": REPETITION_PENALTY,
                                                      "max_new_tokens_compare": MAX_NEW_TOKENS},
        "dense_model": DENSE_MODEL_ID, "reranker_model": RERANKER_MODEL_ID, "retrieval": "condición C del parcial",
        "evidence_score_threshold": EVIDENCE_SCORE_THRESHOLD, "limits": limits.__dict__,
        "sha256": {**{f"data/{n}": sha256_file(DATA_DIR / n) for n in
                      ["corpus.jsonl", "queries.jsonl", "qrels.json", "tasks.jsonl", "expected_results.jsonl",
                       "memory_scenarios.jsonl"]},
                   **{f"prompts/{p.name}": sha256_file(p) for p in sorted(PROMPTS_DIR.glob("*.txt"))}},
        **(extra or {}),
    }


def run_exp1(policies: list, tools, limits: Limits, log=print) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    tasks = read_jsonl(DATA_DIR / "tasks.jsonl")
    expected = {e["task_id"]: e for e in read_jsonl(DATA_DIR / "expected_results.jsonl")}
    out = RESULTS_DIR / "exp1"
    results, rows = [], []
    for policy in policies:
        for task in tasks:
            r = run_task(task, policy, tools, limits, run_label=f"exp1/{policy.name}")
            results.append(r)
            rows.append(eval_exp1(r, expected[task["task_id"]]))
            log(f"{policy.name:12s} {task['task_id']} -> {r['stop_reason']:15s} steps={r['steps']} "
                f"calls={r['tool_calls']} {r['latency_ms']/1000:.1f}s")
    per_task = pd.DataFrame(rows)
    summary = summarize_exp1(per_task)
    per_task.to_csv(out / "per_task.csv", index=False)
    summary.to_csv(out / "summary.csv")
    _jsonl(out / "results.jsonl", [{k: v for k, v in r.items() if k != "trace"} for r in results])
    _jsonl(RESULTS_DIR / "traces" / "exp1_traces.jsonl", [s for r in results for s in r["trace"]])
    (out / "run_config.json").write_text(json.dumps(run_config(limits, {"experiment": "exp1",
                                         "policies": [p.name for p in policies]}), indent=1))
    return per_task, summary, results


def run_exp2(agent, tools, retr, gen, documents, passages_by_id, limits: Limits,
             configs=("M0", "M1", "M2", "M3"), log=print):
    scenarios = read_jsonl(DATA_DIR / "memory_scenarios.jsonl")
    out = RESULTS_DIR / "exp2"
    store = out / "memory_store"
    results, rows, writes = [], [], []
    for cfg in configs:
        mem = make_memory(cfg, store_dir=store, passages_by_id=passages_by_id, count_tokens=gen.count_tokens,
                          encoder=retr, documents=documents)
        current = None
        for item in scenarios:
            if item["session_id"] != current:
                current = item["session_id"]
                mem.start_session(current, item["as_of"])
            r = run_task(item, agent, tools, limits, memory=mem, run_label=f"exp2/{cfg}")
            r["config"] = cfg
            written = mem.write(item, r)
            writes += [{"config": cfg, "after_task": item["task_id"], **w} for w in written]
            results.append(r)
            row = eval_memory(r, item)
            row["config"] = cfg
            rows.append(row)
            log(f"{cfg} {item['task_id']} ({item['session_id']}) -> {r['stop_reason']:15s} "
                f"success={row['task_success']} mem={r['memory_ids_in_context']} cited={row.get('cited_id_proposed')}")
    per = pd.DataFrame(rows)
    summary = summarize_memory(per)
    per.to_csv(out / "per_interaction.csv", index=False)
    summary.to_csv(out / "summary.csv")
    _jsonl(out / "results.jsonl", [{k: v for k, v in r.items() if k != "trace"} for r in results])
    _jsonl(out / "memory_writes.jsonl", writes)
    _jsonl(RESULTS_DIR / "traces" / "exp2_traces.jsonl", [s for r in results for s in r["trace"]])
    (out / "run_config.json").write_text(json.dumps(run_config(limits, {"experiment": "exp2",
                                         "configs": list(configs)}), indent=1))
    return per, summary, results
