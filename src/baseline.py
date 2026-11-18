"""Componentes heredados del examen parcial (sin cambios de comportamiento).

Origen: repositorio del parcial, `parcial/notebooks/Examen_Parcial_CC0F4.ipynb`
  - §1  configuración (modelos, recuperación, decoding)
  - §2  contratos `StructuredQuery`, `RAGAnswer`, `semantic_check`
  - §3  carga y validación del benchmark
  - §4  `build_chunks`
  - §6  recuperadores (E5 + FAISS, BM25, RRF, Cross-Encoder) -> condición C
  - §8  `build_context`, `parse_json_object`, `validate_output`, `Generator`
  - §10 herramienta `citation.check_temporal_validity` con su política de retry

Lo único nuevo aquí es `DATA_DIR` (el corpus se copió al repositorio de la PC3
y se verifica con las mismas huellas SHA-256 del parcial) y que el filtrado
por fecha (`as_of`) se hace fuera de esta clase, en `tools.py`.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import asyncio
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"

# Huellas del parcial (parcial/results/exp1/run_config.json): prueban que es el mismo corpus
PARCIAL_SHA256 = {
    "corpus.jsonl": "a588b591264c4aaeb405424919ea01c40b1de39edbc71f736bf69c97f347e79c",
    "queries.jsonl": "3db358757d5eba8c027b45fb49b448e50a0e7a7803321834bfa325c7ebdee9d5",
    "qrels.json": "6aefedf19b3458c78d2a64b906c67447f958d39cae5a1b0741a3f5cbd21a1636",
}

SEED = 42
DENSE_MODEL_ID = "intfloat/multilingual-e5-small"
RERANKER_MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
GENERATOR_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"       # analista (compare_evidence), igual que el parcial
# Política del agente W1 (nuevo en la PC3). Qwen 0.5B se probó primero en desarrollo y no
# lograba elegir check_citation (results/dev/); ver README §5.
AGENT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
AGENT_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DEVICE = "cpu"

TARGET_WORDS = 100
OVERLAP_PASSAGES = 1
CANDIDATE_DEPTH = 10
RRF_CONSTANT = 60
BM25_K1, BM25_B, BM25_EPSILON = 1.5, 0.75, 0.25
RAG_TOP_K = 3
MAX_NEW_TOKENS = 220
DO_SAMPLE = False
REPETITION_PENALTY = 1.0  # corrección documentada en el parcial (greedy puro)

BM25_PRESERVED_TERMS = frozenset({"no", "not", "without", "more", "less", "than", "always", "only"})
ENGLISH_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "has", "have", "how", "in", "is", "it", "its", "of", "on", "or",
    "such", "that", "the", "their", "them", "these", "this", "to", "was", "we",
    "what", "when", "which", "while", "with", "should", "instead", "much",
}) - BM25_PRESERVED_TERMS

TOPICS = {
    "scaling_laws", "emergence", "chain_of_thought", "rag_vs_finetuning",
    "attention_heads", "alignment", "quantization", "positional_encoding",
    "efficient_attention", "dense_vs_sparse_retrieval", "in_context_learning", "peft",
}


# ---------------------------------------------------------------- contratos (§2)
class Difficulty(str, Enum):
    CLEAR = "clear"
    AMBIGUOUS = "ambiguous"
    HARD = "hard"
    DISTRACTOR = "distractor"
    NO_ANSWER = "no_answer"


class StructuredQuery(BaseModel):
    query_id: str = Field(pattern=r"^Q\d{2}$")
    query: str = Field(min_length=15)
    query_es: str = Field(min_length=10)
    topic: str
    claim_under_test: str = Field(min_length=10)
    difficulty: Difficulty

    @field_validator("topic")
    @classmethod
    def topic_in_domain(cls, value: str) -> str:
        if value not in TOPICS:
            raise ValueError(f"topic fuera del dominio: {value}")
        return value


class RAGRelation(str, Enum):
    AGREE = "agree"
    DISAGREE = "disagree"
    NEUTRAL = "neutral"
    INSUFFICIENT = "insufficient_evidence"


class RAGAnswer(BaseModel):
    relation: RAGRelation
    answer: str = Field(min_length=5)
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


def semantic_check(answer: RAGAnswer, context_passage_ids: set[str]) -> list[str]:
    problems = []
    outside = [c for c in answer.citations if c not in context_passage_ids]
    if outside:
        problems.append(f"citas fuera del contexto: {outside}")
    if answer.relation != RAGRelation.INSUFFICIENT and context_passage_ids and not answer.citations:
        problems.append("afirma una relación sin citar evidencia")
    return problems


# ---------------------------------------------------------------- datos (§3)
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_same_corpus_as_parcial() -> dict[str, str]:
    found = {name: sha256_file(DATA_DIR / name) for name in PARCIAL_SHA256}
    for name, expected in PARCIAL_SHA256.items():
        if found[name] != expected:
            raise ValueError(f"{name} no coincide con el parcial: el benchmark cambió")
    return found


def load_benchmark():
    documents = read_jsonl(DATA_DIR / "corpus.jsonl")
    queries = [StructuredQuery.model_validate(q) for q in read_jsonl(DATA_DIR / "queries.jsonl")]
    qrels = json.loads((DATA_DIR / "qrels.json").read_text())
    references = {r["query_id"]: r for r in read_jsonl(DATA_DIR / "reference_answers.jsonl")}
    raw = json.loads((DATA_DIR / "raw" / "arxiv_raw.json").read_text())
    published = {p["arxiv_id"]: p["published"][:10] for p in raw["papers"]}
    passages_by_id = {
        p["passage_id"]: {**p, "doc_id": d["doc_id"], "title": d["title"], "year": d["year"],
                          "published": published[d["doc_id"]], "arxiv_version": d["arxiv_version"]}
        for d in documents for p in d["passages"]
    }
    return documents, queries, qrels, references, raw, passages_by_id


# ---------------------------------------------------------------- chunks (§4)
def build_chunks(documents: list[dict], target_words: int = TARGET_WORDS,
                 overlap_passages: int = OVERLAP_PASSAGES) -> list[dict]:
    chunks = []
    for doc in documents:
        passages = doc["passages"]
        start, chunk_number = 0, 1
        while start < len(passages):
            selected, total_words, end = [], 0, start
            while end < len(passages):
                n_words = len(passages[end]["text"].split())
                if selected and total_words + n_words > target_words:
                    break
                selected.append(passages[end])
                total_words += n_words
                end += 1
            chunks.append({
                "chunk_id": f"{doc['doc_id']}-C{chunk_number:03d}", "doc_id": doc["doc_id"],
                "title": doc["title"], "year": doc["year"],
                "passage_ids": [p["passage_id"] for p in selected], "n_words": total_words,
                "text": " ".join(p["text"] for p in selected),
            })
            if end >= len(passages):
                break
            start = max(start + 1, end - overlap_passages)
            chunk_number += 1
    return chunks


# ---------------------------------------------------------------- recuperadores (§6)
def lexical_tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)
    return [t for t in tokens if t not in ENGLISH_STOPWORDS]


def stable_order(scores: np.ndarray, items: list[dict], depth: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), items[i]["chunk_id"]))[:depth]


def reciprocal_rank_fusion(rankings, depth: int = CANDIDATE_DEPTH, rank_constant: int = RRF_CONSTANT):
    scores, payload, sources = defaultdict(float), {}, defaultdict(list)
    for ranking in rankings:
        for item in ranking:
            cid = item["chunk_id"]
            scores[cid] += 1.0 / (rank_constant + item["rank"])
            payload[cid] = {k: v for k, v in item.items() if k not in {"score", "rank", "source"}}
            sources[cid].append(item["source"])
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))[:depth]
    return [{**payload[cid], "score": float(scores[cid]), "rank": r,
             "source": "+".join(sorted(set(sources[cid])))} for r, cid in enumerate(ordered, start=1)]


class Retrievers:
    """Condición C del parcial: E5 + FAISS, BM25, RRF k=60 y Cross-Encoder."""

    def __init__(self, chunks: list[dict]):
        import faiss
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.bm25 = BM25Okapi([lexical_tokenize(t) for t in texts], k1=BM25_K1, b=BM25_B, epsilon=BM25_EPSILON)
        self.encoder = SentenceTransformer(DENSE_MODEL_ID, device=DEVICE)
        vectors = self.encoder.encode([f"passage: {t}" for t in texts], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.dim = vectors.shape[1]
        self.reranker = CrossEncoder(RERANKER_MODEL_ID, max_length=512, device=DEVICE)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encoder.encode([f"query: {text}"], normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False).astype("float32")

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode([f"passage: {t}" for t in texts], normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False).astype("float32")

    def _items(self, order, scores, source):
        return [{**self.chunks[i], "score": float(scores[i]), "rank": r, "source": source}
                for r, i in enumerate(order, start=1)]

    def bm25_ranking(self, query: str, allowed: set[int] | None = None, depth: int = CANDIDATE_DEPTH):
        scores = np.asarray(self.bm25.get_scores(lexical_tokenize(query)), dtype=float)
        if allowed is not None:
            scores = np.where([i in allowed for i in range(len(scores))], scores, -np.inf)
        return self._items(stable_order(scores, self.chunks, depth), scores, "bm25")

    def dense_ranking(self, query: str, allowed: set[int] | None = None, depth: int = CANDIDATE_DEPTH):
        scores, idx = self.index.search(self.encode_query(query), int(self.index.ntotal))
        full = np.full(len(self.chunks), -np.inf)
        full[idx[0]] = scores[0]
        if allowed is not None:
            full = np.where([i in allowed for i in range(len(full))], full, -np.inf)
        return self._items(stable_order(full, self.chunks, depth), full, "dense")

    def rerank(self, query: str, candidates: list[dict]):
        if not candidates:
            return []
        scores = np.asarray(self.reranker.predict([(query, c["text"]) for c in candidates],
                                                  batch_size=8, show_progress_bar=False), dtype=float).reshape(-1)
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]["chunk_id"]))
        return [{**candidates[i], "pre_rerank_rank": candidates[i]["rank"], "score": float(scores[i]),
                 "rank": r, "source": "reranker"} for r, i in enumerate(order, start=1)]

    def condition_c(self, query: str, allowed: set[int] | None = None) -> list[dict]:
        dense = [x for x in self.dense_ranking(query, allowed) if np.isfinite(x["score"])]
        bm25 = [x for x in self.bm25_ranking(query, allowed) if np.isfinite(x["score"])]
        return self.rerank(query, reciprocal_rank_fusion([dense, bm25]))


# ---------------------------------------------------------------- generación (§8)
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def parse_json_object(text: str):
    text = text.strip()
    try:
        value = json.loads(text)
        return (value, "raw_json") if isinstance(value, dict) else (None, "unparseable")
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                if isinstance(value, dict):
                    return value, "recovered_json"
            except json.JSONDecodeError:
                pass
    return None, "unparseable"


def validate_output(raw_output: str, context_ids: list[str]) -> dict:
    obj, level = parse_json_object(raw_output)
    result = {"parse_level": level, "schema_valid": False, "semantic_problems": [], "parsed": obj}
    if obj is None:
        return result
    try:
        answer = RAGAnswer.model_validate(obj)
    except ValidationError as exc:
        result["schema_error"] = str(exc).splitlines()[0]
        return result
    result.update(schema_valid=True, parsed=answer.model_dump(mode="json"),
                  semantic_problems=semantic_check(answer, set(context_ids)))
    return result


class Generator:
    """Qwen2.5-Instruct en fp32 y greedy puro (decoding igual al parcial)."""

    def __init__(self, model_id: str = GENERATOR_MODEL_ID, revision: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        torch.manual_seed(SEED)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                          dtype=torch.float32).to(DEVICE).eval()
        self.context_window = self.model.config.max_position_embeddings

    def chat_prompt(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text)["input_ids"])

    def generate_raw(self, prompt: str, max_new_tokens: int, stop_at_newline: bool = False) -> dict:
        torch = self.torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        n_prompt = int(inputs["input_ids"].shape[1])
        start = time.perf_counter()
        kwargs = {}
        if stop_at_newline:
            kwargs["stop_strings"] = ["\n"]
            kwargs["tokenizer"] = self.tokenizer
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=DO_SAMPLE,
                                      temperature=None, top_p=None, top_k=None,
                                      repetition_penalty=REPETITION_PENALTY,
                                      pad_token_id=self.tokenizer.eos_token_id, **kwargs)
        new_tokens = out[0][n_prompt:]
        return {"raw_output": self.tokenizer.decode(new_tokens, skip_special_tokens=True),
                "prompt_tokens": n_prompt, "generated_tokens": int(len(new_tokens)),
                "generation_ms": (time.perf_counter() - start) * 1000.0}

    def generate(self, system: str, user: str, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
        return self.generate_raw(self.chat_prompt(system, user), max_new_tokens)

    def score_options(self, prompt: str, options: list[str]) -> dict:
        """Log-probabilidad (suma) de cada opción como continuación del prompt.

        Es la forma de *decodificación restringida*: el modelo solo puede elegir
        entre las opciones de la allowlist; no genera texto libre.
        """
        torch = self.torch
        start = time.perf_counter()
        prompt_ids = self.tokenizer(prompt)["input_ids"]
        with torch.no_grad():
            out = self.model(torch.tensor([prompt_ids]), use_cache=True)
            past, last_logits = out.past_key_values, out.logits[0, -1]
            scores = {}
            for opt in options:
                opt_ids = self.tokenizer(opt, add_special_tokens=False)["input_ids"]
                logp = torch.log_softmax(last_logits, -1)[opt_ids[0]].item()
                if len(opt_ids) > 1:
                    import copy
                    o = self.model(torch.tensor([opt_ids[:-1]]), past_key_values=copy.deepcopy(past), use_cache=True)
                    lps = torch.log_softmax(o.logits[0], -1)
                    logp += sum(lps[i, opt_ids[i + 1]].item() for i in range(len(opt_ids) - 1))
                scores[opt] = logp
        return {"scores": scores, "prompt_tokens": len(prompt_ids),
                "generation_ms": (time.perf_counter() - start) * 1000.0}


def build_context(passage_ids: list[str], passages_by_id: dict) -> str:
    blocks = []
    for pid in passage_ids:
        p = passages_by_id[pid]
        blocks.append(f'[{pid}] ({p["title"]}, {p["year"]})\n{p["text"]}')
    return "\n\n".join(blocks)


# ---------------------------------------------------------------- herramienta (§10)
ARXIV_ID_PATTERN = r"^\d{4}\.\d{4,5}$"
ErrorKind = Literal["protocol", "validation", "domain", "transient", "timeout", "execution", "output_contract"]


class CheckCitationInput(BaseModel):
    model_config = {"extra": "forbid"}
    citing_id: str = Field(pattern=ARXIV_ID_PATTERN)
    cited_id: str = Field(pattern=ARXIV_ID_PATTERN)


class CheckCitationOutput(BaseModel):
    citing_id: str
    cited_id: str
    citing_date: date
    cited_date: date
    days_between: int
    temporally_valid: bool

    @model_validator(mode="after")
    def consistent(self):
        if self.temporally_valid != (self.cited_date <= self.citing_date):
            raise ValueError("temporally_valid inconsistente con las fechas")
        return self


class ToolError(BaseModel):
    code: str
    kind: ErrorKind
    retryable: bool
    message: str


class ToolResult(BaseModel):
    status: Literal["ok", "error"]
    data: dict[str, Any] | None = None
    error: ToolError | None = None


def error_result(kind, code, message, retryable) -> ToolResult:
    return ToolResult(status="error", error=ToolError(code=code, kind=kind, retryable=retryable, message=message))


class DomainError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class TransientToolError(Exception):
    pass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff_ms: tuple[int, ...] = (20, 40)
    per_attempt_timeout_ms: int = 100
    overall_deadline_ms: int = 400


def exception_to_tool_error(exc: BaseException) -> ToolError:
    if isinstance(exc, DomainError):
        return ToolError(code=exc.code, kind="domain", retryable=False, message=str(exc))
    if isinstance(exc, TransientToolError):
        return ToolError(code="temporary_unavailable", kind="transient", retryable=True, message=str(exc))
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return ToolError(code="attempt_timeout", kind="timeout", retryable=True, message="Intento excedió el timeout.")
    return ToolError(code="execution_error", kind="execution", retryable=False, message=repr(exc))


def should_retry(error: ToolError, attempt: int, max_attempts: int) -> bool:
    return error.retryable and attempt < max_attempts


class CitationTool:
    """`citation.check_temporal_validity` del parcial (§10), mismo contrato y política."""

    name = "citation.check_temporal_validity"

    def __init__(self, raw: dict, policy: RetryPolicy = RetryPolicy()):
        self.catalog = {p["arxiv_id"]: date.fromisoformat(p["published"][:10]) for p in raw["papers"]}
        self.policy = policy

    def _validate_domain(self, args: CheckCitationInput) -> None:
        if args.citing_id == args.cited_id:
            raise DomainError("self_citation", "Un artículo no puede citarse a sí mismo en esta verificación.")
        for field in ("citing_id", "cited_id"):
            if getattr(args, field) not in self.catalog:
                raise DomainError("unknown_paper", f"{field}={getattr(args, field)} no existe en el catálogo local.")

    async def _impl(self, args: CheckCitationInput, fault: str, attempt: int) -> dict:
        if fault == "transient_once" and attempt == 1:
            raise TransientToolError("Catálogo temporalmente no disponible.")
        if fault == "slow_always" or (fault == "slow_once" and attempt == 1):
            await asyncio.sleep(0.5)
        if fault == "bad_output":
            return {"citing_id": args.citing_id, "valid": "unknown"}
        citing, cited = self.catalog[args.citing_id], self.catalog[args.cited_id]
        return {"citing_id": args.citing_id, "cited_id": args.cited_id, "citing_date": citing.isoformat(),
                "cited_date": cited.isoformat(), "days_between": (citing - cited).days,
                "temporally_valid": cited <= citing}

    async def _execute(self, raw_args: dict, fault: str) -> dict:
        policy, started, traces = self.policy, time.perf_counter(), []

        def finish(result: ToolResult, attempts: int) -> dict:
            return {"result": result.model_dump(mode="json"), "attempts": attempts,
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 2), "traces": traces}

        try:
            args = CheckCitationInput.model_validate(raw_args)
        except ValidationError as exc:
            err = exc.errors()[0]
            return finish(error_result("validation", "invalid_arguments",
                                       f"{'.'.join(map(str, err['loc']))}: {err['msg']}", False), 0)
        try:
            self._validate_domain(args)
        except DomainError as exc:
            return finish(ToolResult(status="error", error=exception_to_tool_error(exc)), 0)
        attempt = 0
        while True:
            attempt += 1
            t0 = time.perf_counter()
            remaining = max(0.0, policy.overall_deadline_ms - (time.perf_counter() - started) * 1000.0)
            if remaining <= 0:
                return finish(error_result("timeout", "deadline_exceeded", "Deadline total agotado.", True), attempt - 1)
            try:
                data = await asyncio.wait_for(self._impl(args, fault, attempt),
                                              timeout=min(policy.per_attempt_timeout_ms, remaining) / 1000.0)
            except Exception as exc:
                error = exception_to_tool_error(exc)
                traces.append({"attempt": attempt, "status": "error", "error_kind": error.kind,
                               "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)})
                if not should_retry(error, attempt, policy.max_attempts):
                    return finish(ToolResult(status="error", error=error), attempt)
                backoff = policy.backoff_ms[min(attempt - 1, len(policy.backoff_ms) - 1)]
                left = max(0.0, policy.overall_deadline_ms - (time.perf_counter() - started) * 1000.0)
                if left <= backoff:
                    return finish(error_result("timeout", "deadline_exceeded",
                                               "No queda presupuesto para backoff y otro intento.", True), attempt)
                await asyncio.sleep(backoff / 1000.0)
                continue
            try:
                data = CheckCitationOutput.model_validate(data).model_dump(mode="json")
            except ValidationError:
                traces.append({"attempt": attempt, "status": "error", "error_kind": "output_contract",
                               "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)})
                return finish(error_result("output_contract", "invalid_tool_output",
                                           "La salida viola el contrato.", False), attempt)
            traces.append({"attempt": attempt, "status": "ok", "error_kind": None,
                           "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)})
            return finish(ToolResult(status="ok", data=data), attempt)

    def __call__(self, raw_args: dict, fault: str = "none") -> dict:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._execute(raw_args, fault))
        # Dentro de Jupyter ya hay un loop: se ejecuta en un hilo aparte
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            return ex.submit(asyncio.run, self._execute(raw_args, fault)).result()
