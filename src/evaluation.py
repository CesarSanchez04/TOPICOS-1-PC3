"""Métricas de los experimentos 1 y 2 (denominadores explícitos en cada función)."""
from __future__ import annotations

import pandas as pd


# ------------------------------------------------------------------ Experimento 1
def eval_exp1(result: dict, exp: dict) -> dict:
    """Compara una ejecución con su resultado esperado (fijado antes de ejecutar)."""
    final = result["final_answer"] or {}
    stop = result["stop_reason"]
    termination_correct = stop in exp["acceptable_stops"]
    relation_correct = evidence_supported = citation_correct = None
    if exp["expected_stop"] == "SUCCESS":
        relation_correct = final.get("relation") == exp["gold_relation"]
        evidence_supported = bool(set(final.get("evidence", [])) & set(exp["relevant_passages"]))
        if exp["expected_citation"]:
            c, e = final.get("citation_check") or {}, exp["expected_citation"]
            citation_correct = (c.get("citing_id"), c.get("cited_id"), c.get("temporally_valid")) == \
                               (e["citing_id"], e["cited_id"], e["temporally_valid"])
    if exp["expected_stop"] == "SUCCESS":
        success = (stop == "SUCCESS" and relation_correct and evidence_supported
                   and (citation_correct if exp["expected_citation"] else True))
    else:
        success = termination_correct
    # "Cita verificada" = la verificación de la cita se hizo y su veredicto es correcto,
    # aunque la tarea completa falle por otra razón (por ejemplo, la relación).
    st = result["state"]
    cit_done = None
    if exp.get("expected_citation"):
        c, e = st.get("citation") or {}, exp["expected_citation"]
        cit_done = (c.get("citing_id"), c.get("cited_id"), c.get("temporally_valid")) == \
                   (e["citing_id"], e["cited_id"], e["temporally_valid"])
    return {
        "task_id": result["task_id"], "policy": result["policy"], "expected_stop": exp["expected_stop"],
        "stop_reason": stop, "task_success": bool(success), "termination_correct": termination_correct,
        "relation": final.get("relation"), "gold_relation": exp["gold_relation"],
        "relation_correct": relation_correct, "evidence_supported": evidence_supported,
        "citation_correct": citation_correct, "citation_verified_in_state": cit_done,
        "steps": result["steps"], "tool_calls": result["tool_calls"], "tool_errors": result["tool_errors"],
        "repeated_actions": result["repeated_actions"], "latency_ms": result["latency_ms"],
        "tokens_in": result["tokens_in"], "tokens_out": result["tokens_out"], "tokens": result["tokens"],
        "stop_detail": result["stop_detail"],
    }


def summarize_exp1(per_task: pd.DataFrame) -> pd.DataFrame:
    """task_success y termination_correct: denominador = 12 tareas por política.
    relation_correct / evidence_supported: denominador = tareas con SUCCESS esperado (8).
    citation_correct: denominador = tareas con cita y SUCCESS esperado (6)."""
    g = per_task.groupby("policy")
    rate = lambda col: g[col].apply(lambda s: s.dropna().astype(float).mean())
    out = pd.DataFrame({
        "n_tareas": g.size(),
        "task_success": g["task_success"].mean(),
        "termination_correct": g["termination_correct"].mean(),
        "relation_correct": rate("relation_correct"),
        "evidence_supported": rate("evidence_supported"),
        "citation_correct": rate("citation_correct"),
        "steps_media": g["steps"].mean(),
        "tool_calls_media": g["tool_calls"].mean(),
        "tool_errors_total": g["tool_errors"].sum(),
        "repeated_actions_total": g["repeated_actions"].sum(),
        "latency_ms_media": g["latency_ms"].mean(),
        "tokens_media": g["tokens"].mean(),
        "tokens_in_total": g["tokens_in"].sum(),
        "tokens_out_total": g["tokens_out"].sum(),
    })
    return out.round(3)


# ------------------------------------------------------------------ Experimento 2
def last_citation_args(result: dict) -> dict | None:
    """Argumentos del último check_citation que propuso el agente (ejecutado o bloqueado)."""
    calls = [s for s in result["trace"] if s["selected_tool"] == "check_citation" and s["arguments"]]
    return calls[-1]["arguments"] if calls else None


def eval_memory(result: dict, item: dict) -> dict:
    role, exp = item["memory_role"], item["expected"]
    stop, final, st = result["stop_reason"], result["final_answer"] or {}, result["state"]
    mem_items = result.get("memory_items", [])
    docs_in_memory = {d for m in mem_items for d in (m.get("doc_ids") or [])}
    seen_by_tools = {m["doc_id"] for m in st["evidence_meta"].values()}  # vistos por búsqueda en ESTA tarea
    row = {"task_id": item["task_id"], "sequence_id": item["sequence_id"], "session_id": item["session_id"],
           "task_type": item["task_type"], "stop_reason": stop, "steps": result["steps"],
           "tool_calls": result["tool_calls"], "latency_ms": result["latency_ms"],
           "context_tokens": result["tokens_in"], "memory_tokens": result["memory_tokens"],
           "memory_ids_in_context": result["memory_ids_in_context"]}
    if item["task_type"] == "review":
        row["task_success"] = stop == "SUCCESS" and bool(set(final.get("evidence", [])) & set(exp["evidence_any_of"]))
    else:
        c, e = st.get("citation") or {}, exp["citation"]
        row["task_success"] = stop == "SUCCESS" and (c.get("citing_id"), c.get("cited_id"), c.get("temporally_valid")) \
            == (e["citing_id"], e["cited_id"], e["temporally_valid"])
    args = last_citation_args(result) or {}
    cited = args.get("cited_id")
    row["cited_id_proposed"] = cited
    if role.get("needs_memory"):
        gold = role["gold_doc_id"]
        gold_in_ctx = gold in docs_in_memory
        row.update(needs_memory=True, cross_session=role["cross_session"], gold_doc_id=gold,
                   gold_memory_in_context=gold_in_ctx,
                   # Recuperación correcta de memoria: el recuerdo estaba en el contexto, el agente
                   # usó su valor y NO lo obtuvo de una búsqueda en esta misma tarea.
                   memory_recall_correct=bool(gold_in_ctx and cited == gold and gold not in seen_by_tools),
                   solved_by_tools=bool(cited == gold and gold in seen_by_tools))
        non_gold_mem_docs = docs_in_memory - {gold, args.get("citing_id")}
        row["wrong_memory_use"] = bool(cited and cited in non_gold_mem_docs and cited not in seen_by_tools)
        stale_docs = {m_doc for m in mem_items if m.get("status") == "superseded" for m_doc in m.get("doc_ids", [])}
        if role.get("stale_doc_id") and role["stale_doc_id"] in docs_in_memory:
            stale_docs.add(role["stale_doc_id"])
        row["stale_in_context"] = bool(stale_docs - {gold})
        row["stale_memory_error"] = bool(cited and cited in (stale_docs - {gold}))
    else:
        row.update(needs_memory=False, cross_session=False, wrong_memory_use=False,
                   stale_in_context=False, stale_memory_error=False)
    row["memory_in_context"] = bool(mem_items)
    return row


def summarize_memory(per: pd.DataFrame) -> pd.DataFrame:
    """Denominadores:
    task_success                 todas las interacciones (11 por configuración)
    memory_recall_accuracy       interacciones que necesitan un recuerdo (6)
    cross_session_retention      interacciones que necesitan un recuerdo de OTRA sesión (3)
    stale_memory_error_rate      interacciones con un recuerdo obsoleto en el contexto
    wrong_memory_use_rate        interacciones con al menos un recuerdo en el contexto
    """
    rows = []
    for cfg, d in per.groupby("config"):
        need = d[d["needs_memory"]]
        cross = need[need["cross_session"]]
        stale = d[d["stale_in_context"]]
        withmem = d[d["memory_in_context"]]
        rows.append({
            "config": cfg, "n": len(d),
            "task_success": d["task_success"].mean(),
            "memory_recall_accuracy": need["memory_recall_correct"].astype(float).mean(),
            "recall_n": f"{int(need['memory_recall_correct'].sum())}/{len(need)}",
            "cross_session_retention": cross["memory_recall_correct"].astype(float).mean(),
            "cross_n": f"{int(cross['memory_recall_correct'].sum())}/{len(cross)}",
            "stale_memory_error_rate": stale["stale_memory_error"].astype(float).mean() if len(stale) else float("nan"),
            "stale_n": f"{int(stale['stale_memory_error'].sum())}/{len(stale)}",
            "wrong_memory_use_rate": withmem["wrong_memory_use"].astype(float).mean() if len(withmem) else float("nan"),
            "wrong_n": f"{int(withmem['wrong_memory_use'].sum())}/{len(withmem)}",
            "solved_by_tools": f"{int(need['solved_by_tools'].sum())}/{len(need)}",
            "context_tokens_media": d["context_tokens"].mean(),
            "memory_tokens_media": d["memory_tokens"].mean(),
            "steps_media": d["steps"].mean(),
            "latency_ms_media": d["latency_ms"].mean(),
        })
    return pd.DataFrame(rows).set_index("config").round(3)
