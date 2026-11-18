"""Memoria del agente: M0, M1, M2 y M3 (definiciones operacionales).

  M0  NoMemory         cada tarea empieza solo con el prompt base y la entrada actual.
  M1  WorkingMemory    resúmenes de las interacciones ANTERIORES de la sesión actual.
                       Vive en RAM; `start_session` la borra.
  M2  EpisodicMemory   registro persistente (JSONL en disco) de episodios: tarea,
                       acciones, observaciones, resultado y errores. Se recuperan los
                       episodios pertinentes con relevancia + recencia + importancia
                       (Park et al., 2023), normalizadas a [0, 1] y sumadas.
  M3  SemanticMemory   hechos consolidados (un hecho por artículo citado como evidencia),
                       persistentes, con procedencia (passage_id), versión (arxiv_version),
                       fecha del corpus (`as_of`) y estado. Se recuperan con E5 + FAISS
                       (los mismos de la Semana 4 / el parcial).

Las cuatro configuraciones usan el mismo agente, herramientas, límites y el mismo
presupuesto de tokens para el bloque de memoria (`Limits.max_memory_tokens`).

Ciclo de vida de un recuerdo:
  escritura     -> `write()` lo hace el CONTROLADOR al terminar cada interacción (no el LLM).
  lectura       -> `context_for()` antes de la interacción; el bloque entra al prompt como DATO.
  actualización -> M3: si el mismo artículo vuelve a usarse, se sube `version` y se refresca `as_of`.
  invalidación  -> M3: cuando una revisión hecha con un corpus más reciente cita un artículo
                   del mismo tema publicado DESPUÉS del `as_of` de un hecho anterior, ese hecho
                   pasa a `superseded` (no se borra: sigue siendo historia, marcada como no vigente).
  eliminación   -> `delete(memory_id)` (por ejemplo, por privacidad o por un error demostrado).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np


def _snippet(text: str, n_words: int = 22) -> str:
    words = text.split()
    return " ".join(words[:n_words]) + (" ..." if len(words) > n_words else "")


def _render(item: dict) -> str:
    head = f"[{item['memory_id']} | {item['kind']} | session {item['session_id']} | corpus as of {item['as_of']}"
    if item.get("status") == "superseded":
        head += f" | SUPERSEDED by {item['superseded_by']}"
    return head + "] " + item["text"]


def _fit(items: list[dict], budget: int, count_tokens) -> tuple[str, list[dict], int]:
    lines, kept = [], []
    for it in items:
        candidate = "\n".join(lines + [_render(it)])
        if count_tokens(candidate) > budget:
            break
        lines.append(_render(it))
        kept.append(it)
    block = "\n".join(lines)
    return block, kept, (count_tokens(block) if block else 0)


def summarize_result(task: dict, result: dict, passages_by_id: dict) -> tuple[str, list[str]]:
    """Resumen determinista de una interacción (lo que se guarda como recuerdo)."""
    st = result["state"]
    final = result["final_answer"] or {}
    cited = final.get("evidence") or []
    # Si no hubo respuesta validada, se guarda lo OBSERVADO, marcado como no validado
    label = "Supporting papers (validated)"
    if not cited and st.get("evidence"):
        cited, label = st["evidence"], "Papers observed (not validated)"
    docs, parts = [], []
    for pid in cited:
        d = passages_by_id[pid]["doc_id"]
        if d in docs:
            continue
        docs.append(d)
        p = passages_by_id[pid]
        parts.append(f"{d} \"{p['title']}\" ({p['year']}): {_snippet(p['text'])}")
        if len(docs) == 3:
            break
    text = f"Task {task['task_id']} ({task['objective_en']}) ended {result['stop_reason']}."
    if parts:
        text += f" {label}: " + " | ".join(parts)
    if st.get("citation"):
        c = st["citation"]
        text += f" Citation checked: {c['citing_id']} -> {c['cited_id']} temporally_valid={c['temporally_valid']}."
        docs += [x for x in (c["citing_id"], c["cited_id"]) if x not in docs]
    if st.get("failures"):
        text += f" Errors: {st['failures']}."
    return text, docs


class NoMemory:
    name = "M0_no_memory"

    def __init__(self, *_, **__):
        self.session_id, self.as_of = None, None

    def start_session(self, session_id: str, as_of: str | None):
        self.session_id, self.as_of = session_id, as_of

    def context_for(self, task: dict, budget: int):
        return "", [], 0

    def write(self, task: dict, result: dict):
        return []

    def delete(self, memory_id: str) -> bool:
        return False


class WorkingMemory(NoMemory):
    name = "M1_working"

    def __init__(self, passages_by_id: dict, count_tokens, **_):
        super().__init__()
        self.pby, self.count_tokens, self.items = passages_by_id, count_tokens, []

    def start_session(self, session_id, as_of):
        super().start_session(session_id, as_of)
        self.items = []  # se reinicia en cada sesión

    def context_for(self, task, budget):
        return _fit(list(reversed(self.items)), budget, self.count_tokens)  # lo más reciente primero

    def write(self, task, result):
        text, docs = summarize_result(task, result, self.pby)
        item = {"memory_id": f"W-{task['task_id']}", "kind": "working", "session_id": self.session_id,
                "as_of": task.get("as_of") or "full", "text": text, "doc_ids": docs, "status": "active",
                "provenance": {"task_id": task["task_id"]}}
        self.items.append(item)
        return [item]

    def delete(self, memory_id):
        n = len(self.items)
        self.items = [i for i in self.items if i["memory_id"] != memory_id]
        return len(self.items) < n


class _Persistent(NoMemory):
    def __init__(self, store_dir: Path, passages_by_id: dict, count_tokens, encoder, **_):
        super().__init__()
        self.path = Path(store_dir) / f"{self.name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pby, self.count_tokens, self.encoder = passages_by_id, count_tokens, encoder

    def reset(self):
        self.path.write_text("")

    def load(self) -> list[dict]:  # siempre se lee del disco: así se prueba la persistencia
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def save(self, items: list[dict]):
        self.path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))

    def delete(self, memory_id):
        items = self.load()
        kept = [i for i in items if i["memory_id"] != memory_id]
        self.save(kept)
        return len(kept) < len(items)

    def _cos(self, query: str, texts: list[str]) -> np.ndarray:
        q = self.encoder.encode_query(query)
        m = self.encoder.encode_passages(texts)
        return (m @ q[0]).astype(float)


def _minmax(x: np.ndarray) -> np.ndarray:
    if len(x) == 0 or float(x.max() - x.min()) < 1e-9:
        return np.ones_like(x)
    return (x - x.min()) / (x.max() - x.min())


class EpisodicMemory(_Persistent):
    name = "M2_episodic"
    TOP_K = 2
    RECENCY_DECAY = 0.8

    def context_for(self, task, budget):
        items = self.load()
        if not items:
            return "", [], 0
        now = max(i["t"] for i in items) + 1
        rel = self._cos(task["question"] + " " + task["input_en"], [i["text"] for i in items])
        rec = np.array([self.RECENCY_DECAY ** (now - i["t"]) for i in items])
        imp = np.array([i["importance"] for i in items], dtype=float)
        score = _minmax(rel) + _minmax(rec) + _minmax(imp)
        order = np.argsort(-score)[: self.TOP_K]
        chosen = []
        for j in order:
            it = dict(items[j])
            it["retrieval"] = {"score": round(float(score[j]), 3), "relevance": round(float(rel[j]), 3),
                               "recency": round(float(rec[j]), 3), "importance": it["importance"]}
            chosen.append(it)
        return _fit(chosen, budget, self.count_tokens)

    def write(self, task, result):
        items = self.load()
        text, docs = summarize_result(task, result, self.pby)
        actions = [s["selected_tool"] for s in result["trace"] if s["selected_tool"]]
        ep = {"memory_id": f"E-{task['task_id']}", "kind": "episodic", "session_id": self.session_id,
              "as_of": task.get("as_of") or "full", "t": len(items), "task_id": task["task_id"],
              "question": task["question"], "actions": actions, "outcome": result["stop_reason"],
              "errors": result["state"]["failures"], "doc_ids": docs, "text": text,
              "importance": 1.0 if result["stop_reason"] == "SUCCESS" else 0.3, "status": "active",
              "provenance": {"task_id": task["task_id"], "evidence": (result["final_answer"] or {}).get("evidence", [])}}
        self.save(items + [ep])
        return [ep]


class SemanticMemory(_Persistent):
    name = "M3_semantic"
    TOP_K = 3

    def __init__(self, store_dir, passages_by_id, count_tokens, encoder, documents=None, **kw):
        super().__init__(store_dir, passages_by_id, count_tokens, encoder, **kw)
        self.topic = {d["doc_id"]: d["topic"] for d in (documents or [])}

    def context_for(self, task, budget):
        import faiss
        items = self.load()
        if not items:
            return "", [], 0
        texts = [f"{i['title']}. {i['statement']}" for i in items]
        vecs = self.encoder.encode_passages(texts)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        q = self.encoder.encode_query(task["question"] + " " + task["input_en"])
        scores, idx = index.search(q, min(self.TOP_K, len(items)))
        chosen = []
        for s, j in zip(scores[0], idx[0]):
            it = dict(items[j])
            it["retrieval"] = {"score": round(float(s), 3)}
            chosen.append(it)
        return _fit(chosen, budget, self.count_tokens)

    def write(self, task, result):
        """Consolida hechos solo de revisiones exitosas; aplica actualización e invalidación."""
        final = result["final_answer"] or {}
        if result["stop_reason"] != "SUCCESS" or not final.get("evidence"):
            return []
        items = {i["memory_id"]: i for i in self.load()}
        as_of = task.get("as_of") or "9999-12-31"
        changed = []
        new_docs = []
        for pid in final["evidence"]:
            p = self.pby[pid]
            if p["doc_id"] not in [d for d, _ in new_docs]:
                new_docs.append((p["doc_id"], pid))
        # 1) invalidación: hechos anteriores del mismo tema, escritos con un corpus donde
        #    el artículo nuevo todavía no existía
        for doc_id, pid in new_docs:
            pub = self.pby[pid]["published"]
            for it in items.values():
                if (it["doc_id"] != doc_id and it["status"] == "active"
                        and self.topic.get(it["doc_id"]) == self.topic.get(doc_id)
                        and it["as_of"] < pub and it["published"] < pub):
                    it.update(status="superseded", superseded_by=doc_id,
                              superseded_on=as_of, superseded_by_task=task["task_id"])
                    changed.append(dict(it))
        # 2) escritura / actualización
        for doc_id, pid in new_docs:
            p = self.pby[pid]
            mid = f"F-{doc_id}"
            text = f"{doc_id} \"{p['title']}\" (published {p['published']}): {_snippet(p['text'])}"
            if mid in items:
                it = items[mid]
                it.update(version=it["version"] + 1, last_seen_as_of=as_of, last_task=task["task_id"])
            else:
                it = items[mid] = {
                    "memory_id": mid, "kind": "semantic", "session_id": self.session_id, "doc_id": doc_id,
                    "title": p["title"], "published": p["published"], "arxiv_version": p["arxiv_version"],
                    "statement": p["text"], "text": text, "source_passage_id": pid, "as_of": as_of,
                    "last_seen_as_of": as_of, "version": 1, "status": "active", "superseded_by": None,
                    "doc_ids": [doc_id], "provenance": {"task_id": task["task_id"], "passage_id": pid}}
            changed.append(dict(it))
        self.save(list(items.values()))
        return changed


def make_memory(config: str, *, store_dir: Path, passages_by_id: dict, count_tokens, encoder, documents):
    cls = {"M0": NoMemory, "M1": WorkingMemory, "M2": EpisodicMemory, "M3": SemanticMemory}[config]
    mem = cls(store_dir=store_dir, passages_by_id=passages_by_id, count_tokens=count_tokens,
              encoder=encoder, documents=documents)
    if isinstance(mem, _Persistent):
        mem.reset()
    return mem
