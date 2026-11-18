"""Herramientas que pueden usar el workflow (W0) y el agente (W1).

Todas se ejecutan a través de `ToolRuntime.execute`, nunca directamente:
  1. allowlist de nombres (`TOOL_NAMES`)
  2. contrato de entrada (Pydantic, `extra="forbid"`)
  3. validación de dominio (IDs existentes, evidencia ya observada)
  4. ejecución con timeout por herramienta y política de reintentos
  5. contrato de salida
  6. observación saneada: solo `status`, `kind`, `code` y datos validados.
     El texto libre de un error NO se pasa al modelo (no se convierte en instrucción).

Herramientas:
  - search_evidence  (reutiliza la condición C del parcial, §6)
  - compare_evidence (reutiliza el RAG del parcial, §8: mismo prompt, modelo y decoding)
  - check_citation   (reutiliza `citation.check_temporal_validity` del parcial, §10)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from baseline import (CitationTool, RetryPolicy, TransientToolError, build_context, load_prompt,
                      validate_output)

TOOL_NAMES = ("search_evidence", "compare_evidence", "check_citation")
PASSAGE_ID_PATTERN = r"^\d{4}\.\d{4,5}-P\d{2}$"

# Umbral de "hay evidencia" sobre el score del Cross-Encoder.
# Se fijó con consultas de DESARROLLO (data/dev_tasks.jsonl), no con el benchmark:
# preguntas fuera del corpus obtuvieron <= 1.24 y preguntas del corpus >= 2.52.
EVIDENCE_SCORE_THRESHOLD = 2.0
SEARCH_TOP_CHUNKS = 3
MAX_COMPARE_PASSAGES = 8

# Fallo inyectado para la prueba de "salida inválida" (identificado como tal en las trazas)
INJECTED_BAD_OUTPUT = "Sorry, I cannot help with that request."


class SearchInput(BaseModel):
    model_config = {"extra": "forbid"}
    query: str = Field(min_length=8, max_length=300)


class EvidencePassage(BaseModel):
    passage_id: str = Field(pattern=PASSAGE_ID_PATTERN)
    doc_id: str
    title: str
    year: int
    published: date


class SearchOutput(BaseModel):
    passages: list[EvidencePassage] = Field(max_length=12)
    max_score: float
    evidence_found: bool
    as_of: date | None = None


class CompareInput(BaseModel):
    model_config = {"extra": "forbid"}
    question: str = Field(min_length=15)
    passage_ids: list[str] = Field(min_length=1, max_length=MAX_COMPARE_PASSAGES)


@dataclass
class Observation:
    tool: str
    status: str                       # ok | error
    data: dict | None = None
    error_kind: str | None = None     # validation | domain | transient | timeout | output_contract | protocol ...
    error_code: str | None = None
    retryable: bool = False
    attempts: int = 0
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    injected_fault: str = "none"
    detail: dict = field(default_factory=dict)  # solo para la traza; no entra al prompt

    def to_record(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class ToolRuntime:
    """Ejecuta herramientas con contratos. `faults` inyecta fallos controlados por tarea."""

    TIMEOUT_MS = {"search_evidence": 5_000, "compare_evidence": 60_000, "check_citation": 400}
    RETRY = RetryPolicy()  # misma política del parcial: 3 intentos, backoff 20/40 ms

    def __init__(self, retr, gen, passages_by_id: dict, raw: dict):
        self.retr, self.gen = retr, gen
        self.passages_by_id = passages_by_id
        self.citation = CitationTool(raw)
        self.published = {p["arxiv_id"]: date.fromisoformat(p["published"][:10]) for p in raw["papers"]}
        self.rag_system = load_prompt("rag_system.txt")
        self.rag_user = load_prompt("rag_user.txt")

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _error(tool, kind, code, retryable=False, attempts=0, t0=None, **extra) -> Observation:
        ms = (time.perf_counter() - t0) * 1000.0 if t0 else 0.0
        return Observation(tool=tool, status="error", error_kind=kind, error_code=code,
                           retryable=retryable, attempts=attempts, latency_ms=round(ms, 2), **extra)

    def _with_retry(self, tool: str, fn, t0: float):
        """Reintenta solo errores transitorios o timeouts (igual que la política E del parcial)."""
        attempt = 0
        while True:
            attempt += 1
            a0 = time.perf_counter()
            try:
                result = fn(attempt)
                elapsed = (time.perf_counter() - a0) * 1000.0
                # Timeout cooperativo: una llamada CPU de torch no se puede interrumpir;
                # si excede el límite, el resultado se DESCARTA y cuenta como timeout.
                if elapsed > self.TIMEOUT_MS[tool]:
                    raise TimeoutError(f"{tool} excedió {self.TIMEOUT_MS[tool]} ms")
                return result, attempt, None
            except TransientToolError:
                kind, code = "transient", "temporary_unavailable"
            except TimeoutError:
                kind, code = "timeout", "tool_timeout"
            if attempt >= self.RETRY.max_attempts:
                return None, attempt, (kind, code)
            time.sleep(self.RETRY.backoff_ms[min(attempt - 1, len(self.RETRY.backoff_ms) - 1)] / 1000.0)

    # ------------------------------------------------------------ search_evidence
    def search_evidence(self, args: dict, fault: str = "none", as_of: date | None = None) -> Observation:
        t0 = time.perf_counter()
        try:
            inp = SearchInput.model_validate(args)
        except ValidationError as exc:
            return self._error("search_evidence", "validation", "invalid_arguments", t0=t0,
                               detail={"message": exc.errors()[0]["msg"]})
        allowed = None
        if as_of is not None:
            allowed = {i for i, c in enumerate(self.retr.chunks) if self.published[c["doc_id"]] <= as_of}

        def run(attempt):
            if fault == "transient_once" and attempt == 1:
                raise TransientToolError("índice temporalmente no disponible")
            return self.retr.condition_c(inp.query, allowed)

        ranked, attempts, err = self._with_retry("search_evidence", run, t0)
        if err:
            return self._error("search_evidence", err[0], err[1], retryable=True, attempts=attempts, t0=t0,
                               injected_fault=fault)
        pids = []
        for chunk in ranked[:SEARCH_TOP_CHUNKS]:
            pids += [p for p in chunk["passage_ids"] if p not in pids]
        max_score = float(ranked[0]["score"]) if ranked else float("-inf")
        try:
            out = SearchOutput(
                passages=[EvidencePassage(passage_id=p, doc_id=self.passages_by_id[p]["doc_id"],
                                          title=self.passages_by_id[p]["title"], year=self.passages_by_id[p]["year"],
                                          published=self.passages_by_id[p]["published"]) for p in pids],
                max_score=max_score if ranked else -99.0,
                evidence_found=bool(ranked) and max_score >= EVIDENCE_SCORE_THRESHOLD, as_of=as_of)
        except ValidationError:
            return self._error("search_evidence", "output_contract", "invalid_tool_output", attempts=attempts, t0=t0)
        return Observation(tool="search_evidence", status="ok", data=out.model_dump(mode="json"), attempts=attempts,
                           latency_ms=round((time.perf_counter() - t0) * 1000.0, 2), injected_fault=fault,
                           detail={"chunk_scores": [(c["chunk_id"], round(c["score"], 3)) for c in ranked[:5]]})

    # ------------------------------------------------------------ compare_evidence
    def compare_evidence(self, args: dict, observed_ids: set[str], fault: str = "none") -> Observation:
        t0 = time.perf_counter()
        try:
            inp = CompareInput.model_validate(args)
        except ValidationError as exc:
            return self._error("compare_evidence", "validation", "invalid_arguments", t0=t0,
                               detail={"message": exc.errors()[0]["msg"]})
        unseen = [p for p in inp.passage_ids if p not in observed_ids]
        if unseen:  # el modelo no puede citar evidencia que no fue recuperada en esta tarea
            return self._error("compare_evidence", "domain", "unobserved_evidence", t0=t0, detail={"unseen": unseen})
        context = build_context(inp.passage_ids, self.passages_by_id)
        user = self.rag_user.format(query=inp.question, context=context)
        gen_out = {}

        def run(attempt):
            g = self.gen.generate(self.rag_system, user)
            gen_out.update(g)
            return g

        g, attempts, err = self._with_retry("compare_evidence", run, t0)
        tokens = {"tokens_in": gen_out.get("prompt_tokens", 0), "tokens_out": gen_out.get("generated_tokens", 0)}
        if err:
            return self._error("compare_evidence", err[0], err[1], retryable=True, attempts=attempts, t0=t0, **tokens)
        raw_output = INJECTED_BAD_OUTPUT if fault == "bad_output" else g["raw_output"]
        v = validate_output(raw_output, inp.passage_ids)
        detail = {"raw_output": raw_output, "parse_level": v["parse_level"],
                  "semantic_problems": v["semantic_problems"]}
        if not v["schema_valid"]:
            return self._error("compare_evidence", "output_contract", "invalid_tool_output", attempts=attempts,
                               t0=t0, injected_fault=fault, detail=detail, **tokens)
        parsed = v["parsed"]
        # Validación semántica: solo se conservan citas que estaban en el contexto
        parsed["citations"] = [c for c in parsed["citations"] if c in set(inp.passage_ids)]
        return Observation(tool="compare_evidence", status="ok", data=parsed, attempts=attempts,
                           latency_ms=round((time.perf_counter() - t0) * 1000.0, 2), injected_fault=fault,
                           detail=detail, **tokens)

    # ------------------------------------------------------------ check_citation
    def check_citation(self, args: dict, fault: str = "none") -> Observation:
        t0 = time.perf_counter()
        out = self.citation(args, fault)  # herramienta del parcial, con su propia política async
        res = out["result"]
        if res["status"] == "ok":
            return Observation(tool="check_citation", status="ok", data=res["data"], attempts=out["attempts"],
                               latency_ms=round((time.perf_counter() - t0) * 1000.0, 2), injected_fault=fault,
                               detail={"attempt_traces": out["traces"]})
        e = res["error"]
        return Observation(tool="check_citation", status="error", error_kind=e["kind"], error_code=e["code"],
                           retryable=e["retryable"], attempts=out["attempts"],
                           latency_ms=round((time.perf_counter() - t0) * 1000.0, 2), injected_fault=fault,
                           detail={"attempt_traces": out["traces"]})

    # ------------------------------------------------------------ punto único de entrada
    def execute(self, name: str, args: dict, *, fault: str = "none", as_of: date | None = None,
                observed_ids: set[str] = frozenset()) -> Observation:
        if name not in TOOL_NAMES:
            return self._error(name, "protocol", "unknown_tool")
        if not isinstance(args, dict):
            return self._error(name, "protocol", "malformed_arguments")
        if name == "search_evidence":
            return self.search_evidence(args, fault, as_of)
        if name == "compare_evidence":
            return self.compare_evidence(args, set(observed_ids), fault)
        return self.check_citation(args, fault)


ARXIV_RE = re.compile(r"\d{4}\.\d{4,5}")


def extract_arxiv_id(text: str) -> str | None:
    m = ARXIV_RE.search(text or "")
    return m.group(0) if m else None
