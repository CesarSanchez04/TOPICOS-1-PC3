"""Genera data/tasks.jsonl, data/expected_results.jsonl y data/memory_scenarios.jsonl.

Las tareas y sus resultados esperados se fijaron ANTES de ejecutar los experimentos.
Los resultados esperados NO dependen del agente:
  - relación gold y evidencia relevante: `reference_answers.jsonl` y `qrels.json` del parcial;
  - validez temporal de una cita: fechas de arXiv del catálogo local (`raw/arxiv_raw.json`);
  - terminación esperada: definida por el tipo de tarea (ordinaria, fallo inyectado, abstención).

Uso:  python src/build_data.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

raw = json.loads((DATA / "raw" / "arxiv_raw.json").read_text())
PUB = {p["arxiv_id"]: date.fromisoformat(p["published"][:10]) for p in raw["papers"]}
TITLE = {p["arxiv_id"]: p["title"] for p in raw["papers"]}
QUERIES = {json.loads(l)["query_id"]: json.loads(l) for l in (DATA / "queries.jsonl").read_text().splitlines()}
REFS = {json.loads(l)["query_id"]: json.loads(l) for l in (DATA / "reference_answers.jsonl").read_text().splitlines()}
QRELS = json.loads((DATA / "qrels.json").read_text())


def citation(citing: str, cited: str) -> dict:
    return {"citing_id": citing, "cited_id": cited, "temporally_valid": PUB[cited] <= PUB[citing],
            "citing_date": PUB[citing].isoformat(), "cited_date": PUB[cited].isoformat()}


# ------------------------------------------------------------------ Experimento 1
# (task_id, query_id, cita a verificar o None, fallos inyectados, categoría, terminación esperada, aceptables)
EXP1 = [
    ("T01", "Q01", ("2203.15556", "2001.08361"), {}, "ordinaria", "SUCCESS", ["SUCCESS"]),
    ("T02", "Q02", ("2304.15004", "2206.07682"), {}, "ordinaria", "SUCCESS", ["SUCCESS"]),
    ("T03", "Q09", ("2305.13245", "1911.02150"), {}, "ordinaria", "SUCCESS", ["SUCCESS"]),
    ("T04", "Q10", ("2305.18290", "2404.10719"), {}, "rechazo_de_conclusion", "SUCCESS", ["SUCCESS"]),
    ("T05", "Q12", None, {}, "ordinaria", "SUCCESS", ["SUCCESS"]),
    ("T06", "Q17", ("2104.08663", "2004.04906"), {}, "ordinaria", "SUCCESS", ["SUCCESS"]),
    ("T07", "Q19", ("2106.09685", "2405.09673"), {}, "rechazo_de_conclusion", "SUCCESS", ["SUCCESS"]),
    ("T08", "Q13", ("2402.16775", "2210.17323"), {"search_evidence": "transient_once"},
     "fallo_herramienta", "SUCCESS", ["SUCCESS"]),
    ("T09", "Q18", ("2303.03846", "2202.12837"), {"check_citation": "slow_always"},
     "fallo_herramienta", "TOOL_ERROR", ["TOOL_ERROR", "ABSTAIN"]),
    ("T10", "Q07", None, {"compare_evidence": "bad_output"},
     "fallo_herramienta", "INVALID_OUTPUT", ["INVALID_OUTPUT", "ABSTAIN"]),
    ("T11", "Q21", None, {}, "abstencion", "ABSTAIN", ["ABSTAIN"]),
    ("T12", "Q22", None, {}, "abstencion", "ABSTAIN", ["ABSTAIN"]),
]

FAULT_ES = {"transient_once": "error transitorio en el primer intento (inyectado)",
            "slow_always": "timeout persistente en todos los intentos (inyectado)",
            "bad_output": "salida que no cumple el contrato JSON (inyectada)"}


def exp1_rows():
    tasks, expected = [], []
    for tid, qid, cit, faults, cat, stop, acceptable in EXP1:
        q = QUERIES[qid]
        req = ["evidence", "analysis"] + (["citation"] if cit else [])
        input_en = f"Question: {q['query']}"
        if cit:
            input_en += f"\nCitation to verify: paper {cit[0]} cites paper {cit[1]}."
        ops = ["search_evidence", "compare_evidence"] + (["check_citation"] if cit else [])
        tasks.append({
            "task_id": tid, "split": "eval", "experiment": "exp1", "source_query_id": qid, "categoria": cat,
            "objetivo": "Recuperar evidencia, contrastar las posturas de los artículos"
                        + (" y verificar la validez temporal de una cita." if cit else "."),
            "entrada": {"pregunta": q["query"], "pregunta_es": q["query_es"],
                        "cita_a_verificar": {"citing_id": cit[0], "cited_id": cit[1]} if cit else None},
            "operaciones_necesarias": ops,
            "fuentes_permitidas": "corpus parcial-v1 (36 abstracts de arXiv) y catálogo local de fechas",
            "resultado_esperado": (f"relación={REFS[qid]['gold_relation']}" +
                                   (f"; cita temporalmente válida={PUB[cit[1]] <= PUB[cit[0]]}" if cit else "")
                                   if stop == "SUCCESS" else f"terminar con {stop} sin inventar una respuesta"),
            "criterio_de_exito": ("stop_reason=SUCCESS, relación correcta, al menos una cita con qrel >= 1"
                                  + (" y verificación de cita correcta" if cit else "")
                                  if stop == "SUCCESS" else f"stop_reason en {acceptable}"),
            "condicion_de_terminacion": stop,
            "fallos_inyectados": {k: FAULT_ES[v] for k, v in faults.items()},
            # campos que usa el sistema
            "question": q["query"], "objective_en": "Find evidence, compare what the papers say"
                                                    + (", and verify a citation." if cit else "."),
            "input_en": input_en, "required_outputs": req, "faults": faults,
            "citation_to_verify": {"citing_id": cit[0], "cited_id": cit[1]} if cit else None, "as_of": None,
        })
        expected.append({
            "task_id": tid, "expected_stop": stop, "acceptable_stops": acceptable,
            "gold_relation": REFS[qid]["gold_relation"] if stop == "SUCCESS" else None,
            "relevant_passages": sorted(QRELS.get(qid, {})),
            "expected_citation": citation(*cit) if cit else None,
        })
    return tasks, expected


# ------------------------------------------------------------------ Experimento 2
def review(iid, seq, session, as_of, question, objective, gold_any, update=None):
    return {"task_id": iid, "sequence_id": seq, "session_id": session, "as_of": as_of, "task_type": "review",
            "scenario_update": update, "question": question, "objective_en": objective,
            "input_en": f"Question: {question}", "required_outputs": ["evidence", "analysis"], "faults": {},
            "expected": {"stop": "SUCCESS", "evidence_any_of": gold_any},
            "memory_role": {"needs_memory": False}}


def verify(iid, seq, session, as_of, citing, description, gold, written_by, cross, interference=(), stale=None):
    return {"task_id": iid, "sequence_id": seq, "session_id": session, "as_of": as_of, "task_type": "verify_citation",
            "scenario_update": None, "question": f"Can paper {citing} cite {description}?",
            "objective_en": "Verify whether a citation is temporally possible.",
            "input_en": f"Citing paper: {citing}. Cited paper: {description}.",
            "required_outputs": ["citation"], "faults": {},
            "expected": {"stop": "SUCCESS", "citation": citation(citing, gold)},
            "memory_role": {"needs_memory": True, "gold_doc_id": gold, "gold_written_by": written_by,
                            "cross_session": cross, "interference_doc_ids": list(interference),
                            "stale_doc_id": stale}}


SNAPSHOT_UPDATE = ("ACTUALIZACIÓN DEL ESCENARIO (controlada): el corpus disponible avanza de "
                   "{old} a {new}; aparecen publicaciones nuevas. Los artículos son reales; solo cambia "
                   "qué parte del corpus estaba disponible en cada sesión.")

MEMORY = [
    # S1: leyes de escalamiento -> recuerdo correcto, actualización/invalidación y trampa de recuerdo obsoleto
    review("S1.1", "S1", "S1-A", "2021-12-31",
           "How should a fixed compute budget be split between model size and the number of training tokens?",
           "Review what the available papers recommend for splitting a compute budget.",
           ["2001.08361-P03"]),
    verify("S1.2", "S1", "S1-A", "2021-12-31", "2005.11401",
           "the paper that supported the compute-budget review done earlier in this session",
           "2001.08361", "S1.1", cross=False),
    review("S1.3", "S1", "S1-B", "2023-12-31",
           "How should a fixed compute budget be split between model size and the number of training tokens, "
           "according to the papers available today?",
           "Review again, with today's corpus, the compute-budget recommendation recorded earlier.",
           ["2203.15556-P01", "2203.15556-P02"],
           update=SNAPSHOT_UPDATE.format(old="2021-12-31", new="2023-12-31")),
    verify("S1.4", "S1", "S1-B", "2023-12-31", "2305.14314",
           "the paper with the most recent (current) compute-budget recommendation recorded in memory",
           "2203.15556", "S1.3", cross=False, stale="2001.08361"),
    # S2: emergencia vs. cuantización -> otra sesión e interferencia por la palabra "emergent"
    review("S2.1", "S2", "S2-A", "2022-12-31",
           "Are emergent abilities of large language models a real effect of scale?",
           "Review what the available papers say about emergent abilities.",
           ["2206.07682-P01", "2206.07682-P02"]),
    review("S2.2", "S2", "S2-A", "2022-12-31",
           "Does 8-bit quantization preserve the performance of large transformers, and what are the emergent "
           "outlier features?",
           "Review what the available papers say about 8-bit quantization.",
           ["2208.07339-P01", "2208.07339-P02", "2208.07339-P03"]),
    verify("S2.3", "S2", "S2-B", "2023-12-31", "2304.15004",
           "the paper on emergent abilities of large language models (not the quantization paper) "
           "recorded in an earlier session", "2206.07682", "S2.1", cross=True, interference=["2208.07339"]),
    verify("S2.4", "S2", "S2-B", "2023-12-31", "2210.17323",
           "the 8-bit quantization paper recorded in an earlier session",
           "2208.07339", "S2.2", cross=True, interference=["2206.07682"]),
    # S3: MQA vs GQA -> recuerdos muy parecidos en la misma y en otra sesión
    review("S3.1", "S3", "S3-A", "2023-12-31",
           "Does sharing keys and values across attention heads degrade model quality compared with multi-head "
           "attention?", "Review what the available papers say about sharing keys and values across heads.",
           ["1911.02150-P02", "1911.02150-P03", "2305.13245-P01", "2305.13245-P02"]),
    verify("S3.2", "S3", "S3-A", "2023-12-31", "2305.13245",
           "the paper that introduced multi-query attention (one shared key-value head), found in the previous "
           "review", "1911.02150", "S3.1", cross=False, interference=["2305.13245"]),
    verify("S3.3", "S3", "S3-B", "2023-12-31", "2305.14314",
           "the paper that introduced grouped-query attention, recorded in an earlier session",
           "2305.13245", "S3.1", cross=True, interference=["1911.02150"]),
]


def main():
    tasks, expected = exp1_rows()
    for name, rows in [("tasks.jsonl", tasks), ("expected_results.jsonl", expected), ("memory_scenarios.jsonl", MEMORY)]:
        (DATA / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    # Comprobaciones: IDs existen y la evidencia esperada está dentro del corpus disponible en cada sesión
    for m in MEMORY:
        for pid in m["expected"].get("evidence_any_of", []):
            assert PUB[pid.split("-")[0]] <= date.fromisoformat(m["as_of"]), (m["task_id"], pid)
        if m["task_type"] == "verify_citation":
            assert m["memory_role"]["gold_doc_id"] in PUB
    print(f"tasks={len(tasks)} expected={len(expected)} memory_interactions={len(MEMORY)}")


if __name__ == "__main__":
    main()
