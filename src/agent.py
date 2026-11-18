"""W1: agente controlado por el LLM (ciclo ReAct: Thought -> Action -> Observation).

Qué decide el LLM (Qwen2.5-0.5B-Instruct, greedy, el mismo del parcial):
  - el pensamiento (Thought) de cada paso, que queda en la traza;
  - la siguiente acción entre 5 opciones de una allowlist;
  - la consulta de búsqueda y los IDs de la cita a verificar;
  - cuándo terminar (`finish`) o abstenerse (`abstain`).

Qué NO decide el LLM (lo hace el controlador determinista):
  - si la acción se ejecuta (límites, repetición, contratos);
  - el contenido de la observación (validada y resumida como dato);
  - la etiqueta de terminación (`stop_reason`) y la validación de la respuesta final;
  - qué recuerdos entran al prompt (lo decide el módulo de memoria).

La acción se elige por *decodificación restringida*: se calcula la log-probabilidad
de cada nombre de acción permitido y se toma el máximo. Así el modelo no puede
emitir comandos arbitrarios y las probabilidades quedan en la traza como
justificación cuantitativa de la decisión.
"""
from __future__ import annotations

import math
import re
import time

from baseline import load_prompt
from controller import ACTIONS, Decision, Limits, State
from tools import extract_arxiv_id


def _clean_line(text: str) -> str:
    return text.strip().split("\n")[0].strip().strip('"').strip()


class LLMAgent:
    name = "W1_agent"

    def __init__(self, gen):
        self.gen = gen
        self.system = load_prompt("agent_system.txt")
        self.user_t = load_prompt("agent_user.txt")

    # ------------------------------------------------------------ prompt
    def render_user(self, state: State, limits: Limits, max_history: int | None = None) -> str:
        t = state.task
        docs = {}
        for pid in state.evidence:
            m = state.evidence_meta[pid]
            docs.setdefault(m["doc_id"], (m["title"], m["year"], []))[2].append(pid.split("-")[-1])
        if docs:
            ev = f"{len(state.evidence)} passages from {len(docs)} papers (evidence_found={state.evidence_found})"
            ev += "".join(f"\n    {d} \"{title[:70]}\" ({year}) {','.join(ps)}"
                          for d, (title, year, ps) in list(docs.items())[:6])
        else:
            ev = "none yet"
        checklist = []
        if "evidence" in state.required:
            checklist.append("- evidence (search_evidence): " + (
                f"DONE ({len(state.evidence)} passages, evidence_found={state.evidence_found})" if state.evidence
                else (f"FAILED ({state.failures['search_evidence']})" if "search_evidence" in state.failures else "PENDING")))
        if "analysis" in state.required:
            checklist.append("- analysis (compare_evidence): " + (
                f"DONE (relation={state.analysis['relation']})" if state.analysis
                else (f"FAILED ({state.failures['compare_evidence']})" if "compare_evidence" in state.failures else "PENDING")))
        if "citation" in state.required:
            c = state.citation
            checklist.append("- citation check (check_citation): " + (
                f"DONE ({c['citing_id']} -> {c['cited_id']}, temporally_valid={c['temporally_valid']})" if c
                else (f"FAILED ({state.failures['check_citation']})" if "check_citation" in state.failures else "PENDING")))
        hist = state.history if max_history is None else state.history[-max_history:]
        history = "\n".join(f"{h['step']}. {h['action']}: {h['result']}" for h in hist) or "(empty)"
        return self.user_t.format(
            task_id=t["task_id"], objective=t["objective_en"], task_input=t["input_en"],
            step=state.step, max_steps=limits.max_steps, tool_calls=state.tool_calls,
            max_tool_calls=limits.max_tool_calls, evidence=ev, checklist="\n".join(checklist),
            history=history, memory=state.memory_block or "(no memory available)")

    def fit_prompt(self, state: State, limits: Limits) -> tuple[str, dict]:
        """Respeta el presupuesto de contexto recortando primero el historial antiguo."""
        info = {"history_truncated": False}
        for max_hist in (None, 4, 2, 0):
            prompt = self.gen.chat_prompt(self.system, self.render_user(state, limits, max_hist))
            n = self.gen.count_tokens(prompt)
            if n + limits.max_output_tokens_decision <= limits.max_input_tokens:
                info.update(prompt_tokens=n, history_truncated=max_hist is not None)
                return prompt, info
        info.update(prompt_tokens=n, history_truncated=True, over_budget=True)
        return prompt, info

    # ------------------------------------------------------------ decisión
    def decide(self, state: State, limits: Limits) -> Decision:
        t0 = time.perf_counter()
        prompt, info = self.fit_prompt(state, limits)
        if info.get("over_budget"):
            return Decision("?", valid=False, error="prompt excede max_input_tokens", raw=info,
                            latency_ms=(time.perf_counter() - t0) * 1000)
        tokens_in = tokens_out = 0

        # 1) Thought (texto libre, una línea)
        g = self.gen.generate_raw(prompt + "Thought:", limits.max_output_tokens_decision, stop_at_newline=True)
        thought = _clean_line(g["raw_output"])
        tokens_in += g["prompt_tokens"]; tokens_out += g["generated_tokens"]

        # 2) Action (decodificación restringida sobre la allowlist)
        prefix = prompt + "Thought: " + thought + "\nAction:"
        s = self.gen.score_options(prefix, [" " + a for a in ACTIONS])
        scores = {a.strip(): v for a, v in s["scores"].items()}
        z = max(scores.values())
        norm = sum(math.exp(v - z) for v in scores.values())
        probs = {a: round(math.exp(v - z) / norm, 4) for a, v in scores.items()}
        action = max(scores, key=scores.get)
        tokens_in += s["prompt_tokens"]; tokens_out += 1

        # 3) Argumentos (solo los que el agente debe escribir)
        args, error = {}, None
        prefix = prefix + " " + action + "\n"
        if action == "search_evidence":
            g = self.gen.generate_raw(prefix + "Query:", limits.max_output_tokens_args, stop_at_newline=True)
            tokens_in += g["prompt_tokens"]; tokens_out += g["generated_tokens"]
            args = {"query": _clean_line(g["raw_output"])}
        elif action == "compare_evidence":
            # El analista siempre recibe la pregunta de la tarea y la evidencia observada:
            # el agente decide CUÁNDO comparar, no fabrica el contexto.
            args = {"question": state.task["question"], "passage_ids": state.evidence[:8]}
            if not state.evidence:
                error = "compare_evidence sin evidencia observada"
        elif action == "check_citation":
            g1 = self.gen.generate_raw(prefix + "Citing arXiv ID:", limits.max_output_tokens_id, stop_at_newline=True)
            citing = extract_arxiv_id(g1["raw_output"])
            g2 = self.gen.generate_raw(prefix + f"Citing arXiv ID: {citing}\nCited arXiv ID:", limits.max_output_tokens_id,
                                       stop_at_newline=True)
            cited = extract_arxiv_id(g2["raw_output"])
            tokens_in += g1["prompt_tokens"] + g2["prompt_tokens"]
            tokens_out += g1["generated_tokens"] + g2["generated_tokens"]
            args = {"citing_id": citing, "cited_id": cited}
            if not citing or not cited:
                error = f"IDs no válidos: {g1['raw_output']!r} / {g2['raw_output']!r}"

        return Decision(action=action, args=args, thought=thought, action_scores=probs,
                        tokens_in=tokens_in, tokens_out=tokens_out,
                        latency_ms=(time.perf_counter() - t0) * 1000.0, valid=error is None, error=error,
                        raw={"prompt_tokens": info["prompt_tokens"], "history_truncated": info["history_truncated"]})
