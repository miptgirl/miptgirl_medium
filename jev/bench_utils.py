"""Shared benchmark harness for comparing Jev (TypeSafe) against OpenAI models.

Both notebooks in this folder import from here so that the two models are scored by
exactly the same code. The comparison is only meaningful if the metric, the label
normalisation and the prompt content are identical across models - keeping them in
one module is what guarantees that.

Design notes:

- Both models see the same instruction text and the same bare list of label names.
  Jev gets them as `choice` criteria with null descriptions; OpenAI gets them as a
  JSON-schema string enum. Neither side gets extra hand-written label descriptions,
  because giving one model a richer prompt is the easiest way to produce a
  meaningless benchmark.
- Every result is cached to disk keyed by (model, dataset, record id). Re-running a
  notebook costs nothing for rows already done, and a partially-failed run resumes
  instead of re-spending.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------

CONFIG_PATH = "../../config.json"
CACHE_DIR = Path(".cache")

TYPESAFE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"

# 503 (model_unavailable) is not in the published error list but does occur in practice,
# and it is transient - so it is retried alongside the documented 429/529.
RETRYABLE_STATUS = {429, 500, 502, 503, 529}


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load config.json and push every entry into os.environ (repo convention)."""
    with open(path) as f:
        config = json.load(f)
    for key, value in config.items():
        if isinstance(value, str):
            os.environ[key] = value
    return config


# --------------------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------------------


@dataclass
class Prediction:
    """One model's answer for one record."""

    record_id: str
    model: str
    label: str | None
    confidence: float | None = None
    latency_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    probabilities: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "model": self.model,
            "label": self.label,
            "confidence": self.confidence,
            "latency_s": self.latency_s,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
        }


# --------------------------------------------------------------------------------------
# classifiers
# --------------------------------------------------------------------------------------


class JevClassifier:
    """Single-label classification through the TypeSafe `choice` question type."""

    def __init__(self, labels, instructions, api_key=None, model=JEV_MODEL, name=None):
        self.labels = list(labels)
        self.instructions = instructions
        self.model = model
        self.name = name or model
        self.api_key = api_key or os.environ["JEV_API_KEY"]
        # descriptions are deliberately null - the label name is all either model gets
        self.criteria = {label: None for label in self.labels}

    def predict(self, text, max_retries=5, timeout=90) -> Prediction:
        payload = {
            "state": text,
            "model": self.model,
            "questions": {
                "label": {
                    "type": "choice",
                    "instructions": self.instructions,
                    "criteria": self.criteria,
                }
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        started = time.time()
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    TYPESAFE_ENDPOINT, headers=headers, json=payload, timeout=timeout
                )
            except requests.RequestException as exc:  # network blip
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(_backoff(attempt))
                continue

            if response.status_code == 200:
                body = response.json()
                answer = body["answers"]["label"]
                usage = body.get("usage", {})
                return Prediction(
                    record_id="",
                    model=self.name,
                    label=answer.get("choice"),
                    confidence=answer.get("confidence"),
                    probabilities=answer.get("probabilities", {}),
                    latency_s=round(time.time() - started, 2),
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                )

            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code in RETRYABLE_STATUS and attempt < max_retries:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else _backoff(attempt)
                time.sleep(delay)
                continue
            break

        return Prediction(
            record_id="",
            model=self.name,
            label=None,
            latency_s=round(time.time() - started, 2),
            error=last_error,
        )


class OpenAIClassifier:
    """Single-label classification through OpenAI structured outputs (string enum).

    The JSON schema constrains the answer to the label set, which is the closest
    equivalent to Jev's typed `choice` - neither model can return an unparseable or
    out-of-vocabulary answer, so any accuracy gap is about judgement, not formatting.

    `logprobs` is optional because some models reject it (gpt-5-mini returns 403
    "not allowed to request logprobs from this model"). When unavailable, confidence
    is simply None and the confidence comparisons are skipped.
    """

    def __init__(self, labels, instructions, model="gpt-4.1-mini", api_key=None,
                 use_logprobs=True, name=None):
        from openai import OpenAI

        self.labels = list(labels)
        self.instructions = instructions
        self.model = model
        self.name = name or model
        self.use_logprobs = use_logprobs
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"label": {"type": "string", "enum": self.labels}},
                    "required": ["label"],
                    "additionalProperties": False,
                },
            },
        }

    def predict(self, text, max_retries=5, timeout=90) -> Prediction:
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": text},
        ]
        started = time.time()
        last_error = None
        use_logprobs = self.use_logprobs

        for attempt in range(max_retries + 1):
            kwargs = {}
            if use_logprobs:
                kwargs = {"logprobs": True, "top_logprobs": 1}
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=self.response_format,
                    timeout=timeout,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001 - SDK raises a wide range
                message = str(exc)
                last_error = f"{type(exc).__name__}: {message[:200]}"

                # some models refuse logprobs outright - drop them and retry once
                if use_logprobs and "logprobs" in message:
                    use_logprobs = False
                    continue

                if _is_retryable_sdk_error(exc, message) and attempt < max_retries:
                    time.sleep(_backoff(attempt))
                    continue
                break

            choice = response.choices[0]
            label = json.loads(choice.message.content)["label"]
            usage = response.usage
            return Prediction(
                record_id="",
                model=self.name,
                label=label,
                confidence=_logprob_confidence(choice, label),
                latency_s=round(time.time() - started, 2),
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
            )

        return Prediction(
            record_id="",
            model=self.name,
            label=None,
            latency_s=round(time.time() - started, 2),
            error=last_error,
        )


def _is_retryable_sdk_error(exc, message: str) -> bool:
    """Rate limits and server errors are worth retrying; a spent account is not."""
    if "no credits remaining" in message or "insufficient_quota" in message:
        return False
    status = getattr(exc, "status_code", None)
    if status in RETRYABLE_STATUS:
        return True
    return any(token in message for token in ("429", "500", "502", "503", "529", "Timeout"))


def _logprob_confidence(choice, label) -> float | None:
    """Mean token probability over the tokens that spell out the label value.

    A rough but honest confidence: it measures how certain the model was about the
    characters of the answer, which is not the same quantity as Jev's calibrated
    `confidence`. Compare the two as rankings, not as absolute numbers.
    """
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs else None
    if not content:
        return None

    wanted = label.replace("_", "").lower()
    selected = [
        token.logprob
        for token in content
        if token.token.strip(' "_,{}:').lower() and token.token.strip(' "_,{}:').lower() in wanted
    ]
    if not selected:
        selected = [token.logprob for token in content]
    if not selected:
        return None
    return round(sum(math.exp(lp) for lp in selected) / len(selected), 4)


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------


def _backoff(attempt: int) -> float:
    return min(2**attempt, 30) + random.uniform(0, 0.5)


def _cache_path(dataset: str, model: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in model)
    return CACHE_DIR / f"{dataset}__{safe}.jsonl"


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    cached = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # only treat successful rows as cached, so failures are retried on rerun
            if row.get("error") is None and row.get("label") is not None:
                cached[row["record_id"]] = row
    return cached


def run_benchmark(records, classifier, dataset, text_key="text", id_key="record_id",
                  max_workers=8, use_cache=True, progress_every=25):
    """Run one classifier over `records`, returning a list of result dicts.

    `records` is a list of dicts with an id field and a text field. Results already in
    the cache are reused, so a rerun after a failure only pays for what is missing.
    """
    path = _cache_path(dataset, classifier.name)
    cached = _load_cache(path) if use_cache else {}
    todo = [r for r in records if str(r[id_key]) not in cached]

    print(f"[{classifier.name}] {len(records)} records | cached {len(cached)} | to run {len(todo)}")

    results = list(cached.values())
    if not todo:
        return _ordered(results, records, id_key)

    lock = threading.Lock()
    done = 0
    handle = open(path, "a")

    def work(record):
        nonlocal done
        prediction = classifier.predict(record[text_key])
        prediction.record_id = str(record[id_key])
        row = prediction.to_dict()
        with lock:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"  ... {done}/{len(todo)}")
        return row

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for row in pool.map(work, todo):
                results.append(row)
    finally:
        handle.close()

    failures = [r for r in results if r.get("error")]
    if failures:
        print(f"  {len(failures)} failed - e.g. {failures[0]['error'][:120]}")
        print("  rerun this cell to retry only the failures")

    return _ordered(results, records, id_key)


def _ordered(results, records, id_key):
    by_id = {r["record_id"]: r for r in results}
    return [by_id[str(rec[id_key])] for rec in records if str(rec[id_key]) in by_id]


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------


def normalise(label) -> str:
    """Label normalisation applied identically to gold and to every model."""
    if label is None:
        return ""
    return str(label).strip().lower().replace(" ", "_")


def score_predictions(results, gold_by_id):
    """Attach correctness to each result row."""
    scored = []
    for row in results:
        gold = gold_by_id[row["record_id"]]
        predicted = row.get("label")
        scored.append(
            {
                **row,
                "gold": gold,
                "correct": (
                    None if predicted is None else normalise(predicted) == normalise(gold)
                ),
            }
        )
    return scored


def summarise(scored_by_model: dict) -> "list[dict]":
    """One summary row per model: accuracy, coverage, latency and token cost."""
    summary = []
    for model, rows in scored_by_model.items():
        answered = [r for r in rows if r["correct"] is not None]
        correct = [r for r in answered if r["correct"]]
        latencies = sorted(r["latency_s"] for r in rows if r.get("latency_s") is not None)
        summary.append(
            {
                "model": model,
                "n": len(rows),
                "answered": len(answered),
                "accuracy": round(len(correct) / len(answered), 4) if answered else None,
                "errors": len(rows) - len(answered),
                "median_latency_s": latencies[len(latencies) // 2] if latencies else None,
                "input_tokens": sum(r.get("input_tokens") or 0 for r in rows),
                "output_tokens": sum(r.get("output_tokens") or 0 for r in rows),
            }
        )
    return summary


def wilson_interval(correct: int, total: int, z: float = 1.96):
    """95% confidence interval for an accuracy figure.

    On a 200-row test set the sampling error is several points wide, which is usually
    larger than the gap people try to read between two models - so report it.
    """
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


def mcnemar(rows_a, rows_b):
    """Paired test for whether two models differ on the same items.

    Returns (b, c, p_value, n_shared) where b = A right / B wrong, c = A wrong / B right.
    Uses an exact binomial two-sided p-value, which is appropriate at these sample sizes.

    Only items that BOTH models actually answered are counted. A row that errored has
    `correct is None`, which is not the same thing as a wrong answer - counting API
    failures as losses would manufacture a significant result out of an outage.
    """
    a_by_id = {r["record_id"]: r for r in rows_a if r.get("correct") is not None}
    b_by_id = {r["record_id"]: r for r in rows_b if r.get("correct") is not None}
    shared = [i for i in a_by_id if i in b_by_id]

    b = sum(
        1 for i in shared if a_by_id[i]["correct"] and not b_by_id[i]["correct"]
    )
    c = sum(
        1 for i in shared if not a_by_id[i]["correct"] and b_by_id[i]["correct"]
    )

    n = b + c
    if n == 0:
        return b, c, 1.0, len(shared)
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return b, c, round(min(1.0, 2 * tail), 4), len(shared)


# --------------------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------------------

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"


def style_axis(ax):
    """Recessive axes: no chart junk competing with the data."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.set_axisbelow(True)
