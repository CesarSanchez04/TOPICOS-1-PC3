"""Ejecuciones de DESARROLLO (no forman parte de la evaluación).

Sirven para depurar el código, fijar el umbral de evidencia y elegir el modelo de la
política del agente ANTES de correr los experimentos. Usan data/dev_tasks.jsonl y
data/dev_memory_scenarios.jsonl, cuyos temas no se usan en el benchmark de evaluación.

Uso:  python src/dev_runs.py Qwen/Qwen2.5-0.5B-Instruct
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import (DATA_DIR, RESULTS_DIR, Generator, Retrievers, build_chunks, load_benchmark,  # noqa: E402
                      read_jsonl)
from agent import LLMAgent  # noqa: E402
from controller import Limits, run_task  # noqa: E402
from memory import make_memory  # noqa: E402
from tools import ToolRuntime  # noqa: E402


def main(agent_model: str):
    documents, _, _, _, raw, pby = load_benchmark()
    retr = Retrievers(build_chunks(documents))
    analyst = Generator()
    policy_llm = analyst if agent_model == analyst.model_id else Generator(agent_model)
    tools, agent = ToolRuntime(retr, analyst, pby, raw), LLMAgent(policy_llm)
    out = RESULTS_DIR / "dev" / (agent_model.split("/")[-1] + ".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in read_jsonl(DATA_DIR / "dev_tasks.jsonl"):
        r = run_task(t, agent, tools, Limits(), run_label=f"dev/{agent_model}")
        rows.append(r)
        print(t["task_id"], r["stop_reason"], [(s["selected_tool"], s["observation_status"]) for s in r["trace"]])
    mem = make_memory("M2", store_dir=RESULTS_DIR / "dev" / "store", passages_by_id=pby,
                      count_tokens=policy_llm.count_tokens, encoder=retr, documents=documents)
    current = None
    for it in read_jsonl(DATA_DIR / "dev_memory_scenarios.jsonl"):
        if it["session_id"] != current:
            current = it["session_id"]
            mem.start_session(current, it["as_of"])
        r = run_task(it, agent, tools, Limits(), memory=mem, run_label=f"dev/{agent_model}/M2")
        mem.write(it, r)
        rows.append(r)
        print(it["task_id"], r["stop_reason"], [(s["selected_tool"], s["arguments"] if s["selected_tool"] == "check_citation" else "", s["observation_status"]) for s in r["trace"]])
    out.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct")
