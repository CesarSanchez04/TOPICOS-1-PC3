"""W0: workflow determinista (plan fijo, reglas predefinidas, sin LLM para decidir).

Plan:
    entrada validada -> search_evidence(pregunta) -> [abstain si no hay evidencia]
    -> compare_evidence (si se requiere análisis) -> check_citation (si se requiere)
    -> finish (el controlador valida la respuesta final)

Reglas ante errores: el workflow NO reintenta por su cuenta (los reintentos los hace
la política de cada herramienta). Si una herramienta requerida falló, pide `finish`
y el controlador asigna TOOL_ERROR o INVALID_OUTPUT según el tipo de fallo.

El único LLM que interviene en W0 es el analista de `compare_evidence`, el mismo
que usa W1. Así la diferencia entre W0 y W1 es solo la política de control.
"""
from __future__ import annotations

from controller import Decision, Limits, State


class DeterministicWorkflow:
    name = "W0_workflow"

    def decide(self, state: State, limits: Limits) -> Decision:
        task, req = state.task, state.required
        if "search_evidence" in state.failures and not state.evidence:
            return Decision("finish", thought="regla: la búsqueda falló -> terminar y reportar")
        if not state.searches:
            return Decision("search_evidence", {"query": task["question"]}, thought="regla 1: buscar con la pregunta")
        if state.evidence_found is False and "analysis" in req:
            return Decision("abstain", thought=f"regla 2: max_score={state.max_score:.2f} < umbral -> abstenerse")
        if "analysis" in req and state.analysis is None:
            if "compare_evidence" in state.failures:
                return Decision("finish", thought="regla: el análisis falló -> terminar y reportar")
            return Decision("compare_evidence", {"question": task["question"], "passage_ids": state.evidence[:8]},
                            thought="regla 3: analizar la evidencia recuperada")
        if "citation" in req and state.citation is None:
            if "check_citation" in state.failures:
                return Decision("finish", thought="regla: la verificación de cita falló -> terminar y reportar")
            c = task["citation_to_verify"]
            return Decision("check_citation", {"citing_id": c["citing_id"], "cited_id": c["cited_id"]},
                            thought="regla 4: verificar la cita indicada en la tarea")
        return Decision("finish", thought="regla 5: todas las salidas requeridas están disponibles")
