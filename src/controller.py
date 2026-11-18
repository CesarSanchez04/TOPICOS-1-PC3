"""Controlador determinista compartido por W0 (workflow) y W1 (agente).

La POLÍTICA (W0 o W1) solo propone la siguiente acción. El CONTROLADOR decide si
esa acción se ejecuta, la ejecuta mediante `ToolRuntime`, valida la observación,
actualiza el estado, aplica los límites y asigna `stop_reason`.

Definiciones verificables de terminación (se asignan en `run_task`):
  SUCCESS          la política pidió `finish` y la respuesta final pasó la validación
                   con una relación distinta de `insufficient_evidence` (o, en tareas
                   solo de citas, con la verificación de la cita disponible).
  ABSTAIN          la política pidió `abstain`, o el análisis concluyó `insufficient_evidence`.
  MAX_STEPS        se alcanzó `max_steps` o `max_tool_calls` sin acción terminal.
  TIMEOUT          se superó `total_deadline_s` antes de terminar.
  TOOL_ERROR       se pidió `finish` pero falta una salida requerida porque su herramienta
                   falló sin recuperarse (error transitorio agotado, timeout, dominio).
  INVALID_OUTPUT   (a) la política produjo dos decisiones inválidas seguidas, o
                   (b) falta una salida requerida porque la herramienta devolvió una
                       salida que viola su contrato, o
                   (c) `finish` prematuro: falta una salida requerida que nunca se intentó.
  NO_PROGRESS      `no_progress_window` pasos seguidos sin actualizar el estado.
  REPEATED_ACTION  la política repite `max_repeats` veces una acción (mismo nombre y
                   argumentos) que ya se ejecutó; la repetición no se ejecuta.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from tools import Observation, ToolRuntime

TERMINAL_ACTIONS = ("finish", "abstain")
ACTIONS = ("search_evidence", "compare_evidence", "check_citation", "finish", "abstain")


class StopReason(str, Enum):
    SUCCESS = "SUCCESS"
    ABSTAIN = "ABSTAIN"
    MAX_STEPS = "MAX_STEPS"
    TIMEOUT = "TIMEOUT"
    TOOL_ERROR = "TOOL_ERROR"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    NO_PROGRESS = "NO_PROGRESS"
    REPEATED_ACTION = "REPEATED_ACTION"


@dataclass(frozen=True)
class Limits:
    """Fijados ANTES de los experimentos; iguales para W0, W1 y M0-M3."""
    max_steps: int = 8
    max_tool_calls: int = 6
    total_deadline_s: float = 180.0
    timeout_per_tool_ms: tuple = (("search_evidence", 5_000), ("compare_evidence", 60_000), ("check_citation", 400))
    max_input_tokens: int = 1_800          # presupuesto de contexto por llamada al LLM
    max_memory_tokens: int = 350           # parte del presupuesto reservada para memoria
    max_output_tokens_decision: int = 40   # pensamiento del agente
    max_output_tokens_args: int = 32       # consulta de búsqueda generada
    max_output_tokens_id: int = 16         # un arXiv ID (Qwen tokeniza cada dígito)
    max_output_tokens_compare: int = 220   # mismo valor que el parcial
    no_progress_window: int = 3
    max_repeats: int = 2


class FinalAnswer(BaseModel):
    task_id: str
    relation: str | None = None
    answer: str | None = None
    evidence: list[str] = Field(default_factory=list)
    citation_check: dict | None = None
    memory_ids_in_context: list[str] = Field(default_factory=list)


@dataclass
class Decision:
    action: str
    args: dict = field(default_factory=dict)
    thought: str = ""
    action_scores: dict | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    valid: bool = True
    error: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class State:
    task: dict
    as_of: date | None = None
    step: int = 0
    tool_calls: int = 0
    evidence: list[str] = field(default_factory=list)      # passage_ids observados (ordenados)
    evidence_meta: dict = field(default_factory=dict)      # pid -> {doc_id, title, year}
    evidence_found: bool | None = None
    max_score: float | None = None
    searches: list[str] = field(default_factory=list)
    analysis: dict | None = None
    citation: dict | None = None
    failures: dict = field(default_factory=dict)           # tool -> error_kind de su último intento
    history: list[dict] = field(default_factory=list)      # resumen breve para el prompt
    memory_block: str = ""
    memory_items: list[dict] = field(default_factory=list)
    memory_tokens: int = 0

    @property
    def required(self) -> list[str]:
        return self.task["required_outputs"]

    def status_line(self) -> dict:
        return {
            "evidence": f"{len(self.evidence)} passages" if self.evidence else "none",
            "evidence_found": self.evidence_found,
            "analysis": (self.analysis or {}).get("relation", "not done") if "analysis" in self.required else "not required",
            "citation": ("not required" if "citation" not in self.required else
                         ("done" if self.citation else ("failed:" + self.failures["check_citation"]
                                                        if "check_citation" in self.failures else "not done"))),
        }


def signature(action: str, args: dict) -> str:
    return action + json.dumps(args, sort_keys=True)


def build_final(state: State) -> tuple[StopReason, FinalAnswer | None, str]:
    """Valida la respuesta final. Nunca rellena con valores inventados."""
    task, req = state.task, state.required
    tool_for = {"evidence": "search_evidence", "analysis": "compare_evidence", "citation": "check_citation"}
    present = {"evidence": bool(state.evidence), "analysis": state.analysis is not None,
               "citation": state.citation is not None}
    for out in req:
        if not present[out]:
            kind = state.failures.get(tool_for[out])
            if kind is None:
                return StopReason.INVALID_OUTPUT, None, f"finish prematuro: falta '{out}' y nunca se obtuvo"
            if kind == "output_contract":
                return StopReason.INVALID_OUTPUT, None, f"'{out}' no disponible: la herramienta violó su contrato de salida"
            return StopReason.TOOL_ERROR, None, f"'{out}' no disponible: {tool_for[out]} falló ({kind})"
    final = FinalAnswer(task_id=task["task_id"],
                        memory_ids_in_context=[m["memory_id"] for m in state.memory_items])
    if "analysis" in req:
        a = state.analysis
        final.relation, final.answer = a["relation"], a["answer"]
        final.evidence = [c for c in a["citations"] if c in state.evidence]
        if final.relation == "insufficient_evidence":
            return StopReason.ABSTAIN, final, "el análisis concluyó evidencia insuficiente"
        if not final.evidence:
            return StopReason.INVALID_OUTPUT, None, "relación afirmada sin citar evidencia observada"
    elif "evidence" in req:
        final.evidence = list(state.evidence)
    if "citation" in req:
        final.citation_check = state.citation
    return StopReason.SUCCESS, final, "respuesta validada"


def run_task(task: dict, policy, tools: ToolRuntime, limits: Limits = Limits(), *,
             memory=None, run_label: str = "") -> dict:
    """Ejecuta UNA tarea con una política. Devuelve resultado + traza paso a paso."""
    faults = task.get("faults", {})
    as_of = date.fromisoformat(task["as_of"]) if task.get("as_of") else None
    state = State(task=task, as_of=as_of)
    if memory is not None:
        block, items, n_tok = memory.context_for(task, limits.max_memory_tokens)
        state.memory_block, state.memory_items, state.memory_tokens = block, items, n_tok

    trace, executed, started = [], {}, time.perf_counter()
    tokens_in = tokens_out = tool_errors = repeated = invalid_streak = no_progress = 0
    stop, final, why = None, None, ""

    def record(decision: Decision | None, obs: Observation | None, updated: bool, stop_reason=None, note=""):
        trace.append({
            "run": run_label, "task_id": task["task_id"], "policy": getattr(policy, "name", "?"),
            "step": state.step, "thought": decision.thought if decision else "",
            "selected_tool": decision.action if decision else None,
            "arguments": decision.args if decision else {},
            "action_scores": decision.action_scores if decision else None,
            "decision_valid": decision.valid if decision else None,
            "observation_status": (obs.status if obs else ("invalid_decision" if decision and not decision.valid else None)),
            "observation_error": (f"{obs.error_kind}:{obs.error_code}" if obs and obs.status == "error" else None),
            "injected_fault": obs.injected_fault if obs else "none",
            "attempts": obs.attempts if obs else 0,
            "state_updated": updated,
            "tokens": (decision.tokens_in + decision.tokens_out if decision else 0) +
                      (obs.tokens_in + obs.tokens_out if obs else 0),
            "latency_ms": round((decision.latency_ms if decision else 0) + (obs.latency_ms if obs else 0), 2),
            "memory_ids_in_context": [m["memory_id"] for m in state.memory_items],
            "stop_reason": stop_reason.value if stop_reason else None, "note": note,
            "observation_detail": obs.detail if obs else (decision.raw if decision else {}),
        })

    while stop is None:
        if (time.perf_counter() - started) > limits.total_deadline_s:
            stop, why = StopReason.TIMEOUT, "deadline total agotado"
            record(None, None, False, stop, why)
            break
        if state.step >= limits.max_steps:
            stop, why = StopReason.MAX_STEPS, f"max_steps={limits.max_steps}"
            record(None, None, False, stop, why)
            break
        state.step += 1
        decision = policy.decide(state, limits)
        tokens_in += decision.tokens_in
        tokens_out += decision.tokens_out

        if not decision.valid:
            invalid_streak += 1
            no_progress += 1
            state.history.append({"step": state.step, "action": decision.action or "?", "result": "invalid decision"})
            if invalid_streak >= 2:
                stop, why = StopReason.INVALID_OUTPUT, f"dos decisiones inválidas seguidas: {decision.error}"
            elif no_progress >= limits.no_progress_window:
                stop, why = StopReason.NO_PROGRESS, "pasos sin actualizar el estado"
            record(decision, None, False, stop, decision.error or "")
            continue
        invalid_streak = 0

        if decision.action == "abstain":
            stop, why = StopReason.ABSTAIN, "la política decidió abstenerse"
            record(decision, None, False, stop, why)
            break
        if decision.action == "finish":
            stop, final, why = build_final(state)
            record(decision, None, False, stop, why)
            break

        sig = signature(decision.action, decision.args)
        if sig in executed:
            # Una acción idéntica ya ejecutada no se vuelve a ejecutar: se bloquea y se informa
            repeated += 1
            no_progress += 1
            state.history.append({"step": state.step, "action": decision.action,
                                  "result": "blocked: identical action already executed"})
            if repeated >= limits.max_repeats:
                stop, why = StopReason.REPEATED_ACTION, f"acción repetida sin avance: {decision.action}"
            elif no_progress >= limits.no_progress_window:
                stop, why = StopReason.NO_PROGRESS, f"{no_progress} pasos sin actualizar el estado"
            record(decision, None, False, stop, why or "repetición bloqueada")
            continue
        if state.tool_calls >= limits.max_tool_calls:
            stop, why = StopReason.MAX_STEPS, f"max_tool_calls={limits.max_tool_calls}"
            record(decision, None, False, stop, why)
            break

        state.tool_calls += 1
        obs = tools.execute(decision.action, decision.args, fault=faults.get(decision.action, "none"),
                            as_of=as_of, observed_ids=set(state.evidence))
        executed[sig] = executed.get(sig, 0) + 1
        tokens_in += obs.tokens_in
        tokens_out += obs.tokens_out
        updated = False
        if obs.status == "error":
            tool_errors += 1
            state.failures[obs.tool] = obs.error_kind
            # La observación de error se resume como DATO (tipo y código), sin el mensaje libre
            state.history.append({"step": state.step, "action": decision.action,
                                  "result": f"error kind={obs.error_kind} code={obs.error_code}"})
        else:
            state.failures.pop(obs.tool, None)
            if obs.tool == "search_evidence":
                new = [p["passage_id"] for p in obs.data["passages"] if p["passage_id"] not in state.evidence]
                for p in obs.data["passages"]:
                    state.evidence_meta[p["passage_id"]] = {k: p[k] for k in ("doc_id", "title", "year", "published")}
                state.evidence += new
                state.searches.append(decision.args.get("query", ""))
                state.evidence_found = bool(state.evidence_found) or obs.data["evidence_found"]
                state.max_score = max(state.max_score or -99.0, obs.data["max_score"])
                updated = bool(new)
                docs = sorted({p["doc_id"] for p in obs.data["passages"]})
                state.history.append({"step": state.step, "action": "search_evidence",
                                      "result": f"{len(new)} new passages from {len(docs)} papers; "
                                                f"evidence_found={obs.data['evidence_found']}"})
            elif obs.tool == "compare_evidence":
                updated = state.analysis != obs.data
                state.analysis = obs.data
                state.history.append({"step": state.step, "action": "compare_evidence",
                                      "result": f"relation={obs.data['relation']}"})
            elif obs.tool == "check_citation":
                updated = state.citation != obs.data
                state.citation = obs.data
                state.history.append({"step": state.step, "action": "check_citation",
                                      "result": f"temporally_valid={obs.data['temporally_valid']}"})
        no_progress = 0 if updated else no_progress + 1
        if no_progress >= limits.no_progress_window:
            stop, why = StopReason.NO_PROGRESS, f"{no_progress} pasos sin actualizar el estado"
        record(decision, obs, updated, stop, why if stop else "")

    total_ms = (time.perf_counter() - started) * 1000.0
    return {
        "run": run_label, "task_id": task["task_id"], "policy": getattr(policy, "name", "?"),
        "stop_reason": stop.value, "stop_detail": why,
        "final_answer": final.model_dump() if final else None,
        "steps": state.step, "tool_calls": state.tool_calls, "tool_errors": tool_errors,
        "repeated_actions": repeated, "latency_ms": round(total_ms, 1),
        "tokens_in": tokens_in, "tokens_out": tokens_out, "tokens": tokens_in + tokens_out,
        "memory_tokens": state.memory_tokens,
        "memory_ids_in_context": [m["memory_id"] for m in state.memory_items],
        "memory_items": [{k: m.get(k) for k in ("memory_id", "kind", "session_id", "as_of", "doc_ids", "status",
                                                 "superseded_by", "version", "retrieval")}
                         for m in state.memory_items],
        "memory_block": state.memory_block,
        "state": {"evidence": state.evidence, "evidence_meta": state.evidence_meta, "analysis": state.analysis,
                  "citation": state.citation, "failures": state.failures, "searches": state.searches,
                  "evidence_found": state.evidence_found, "max_score": state.max_score},
        "trace": trace,
    }
