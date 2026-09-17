# Jev / TypeSafe API - a POC and three benchmarks

[TypeSafe](https://docs.typesafe.ai/api) exposes a single evaluation endpoint, `POST /v1/systemone`.
You send it some `state` (the content to judge) plus a map of **typed questions**, and get a
structured answer per question. `jev-latest` is the model that serves those requests, so a "Jev
call" is a request to `/v1/systemone` with `"model": "jev-latest"`.

The appeal of the typed-question design is that you never parse free text. You ask a `noul`
(yes/no), a `choice` (pick one) or a `score` (rubric rating), and get back a number or an enum
value **with a confidence attached**. That last part is what the benchmarks below end up being
about.

## Contents

| Notebook | What it does |
| --- | --- |
| `jev_typesafe_poc.ipynb` | API walkthrough - one call per question type, batch eval, usage tracking |
| `benchmark_banking77.ipynb` | Jev vs GPT on 77-intent classification, 1,000 held-out rows |
| `benchmark_banking77_top7.ipynb` | Does a shorter option list help? Same rows, 77 labels vs 7 |
| `benchmark_stackexchange.ipynb` | Jev vs GPT on 14 topics - an easy control for the hard task |

All notebooks are committed with real outputs. Every API call is cached to `cache/` (gitignored)
keyed by model and a hash of the row text, so re-running costs nothing for rows already answered
and a failed run resumes instead of re-spending.

## Setup

```bash
pip install -r requirements.txt
jupyter lab benchmark_banking77.ipynb
```

Keys are read from `../../config.json` (outside the repo, so never committed) under
`JEV_API_KEY` and `OPENAI_API_KEY`. Paths resolve from a `BASE` anchor, so the notebooks work
whether the kernel starts in `jev/` or at the repo root - Jupyter and VS Code disagree about that.

## What the benchmarks found

**Accuracy** - Banking77, 1,000 rows of the official test split, zero-shot, every model
format-constrained to the label set (Jev via `choice` criteria, OpenAI via a JSON-schema enum):

| Model | Banking77 (77 labels) | Stack Exchange (14 topics) |
| --- | --- | --- |
| gpt-5.6-luna | **86.2%** | **96.8%** |
| gpt-5.6-terra | 83.9% | 95.4% |
| jev-latest | 79.0% | 94.6% |

At 1,000 rows all three separate (paired McNemar, worst case p=0.002). At 201 rows none of them
did - the sample size changed the conclusion, not the models.

**The label set is most of Jev's gap.** Running the same 1,001 rows under both a 77-label and a
7-label option list:

| Model | 77 labels | 7 labels | Change |
| --- | --- | --- | --- |
| jev-latest | 83.4% | **97.8%** | **+14.4pp** |
| gpt-5.6-luna | 89.1% | 97.8% | +8.7pp |
| gpt-5.6-terra | 88.8% | 97.9% | +9.1pp |

With 7 options all three converge and Jev's deficit disappears. 96% of its 77-label mistakes were
picks *outside* the 7 true categories. Output tokens drop from 831 to 114 per call, because Jev
returns a probability for every option. Narrowing the candidate list before calling is a real
deployment lever, and it helps Jev roughly twice as much as it helps the GPT models.

**Latency** - 100 sequential uncached calls per model:

| Model | median | p90 | max |
| --- | --- | --- | --- |
| jev-latest | **0.67s** | 0.79s | 1.18s |
| gpt-5.6-luna | 1.01s | 2.01s | 3.92s |
| gpt-5.6-terra | 1.03s | 1.40s | - |

Jev's entire range sits inside the other two models' interquartile boxes - it is the only one
whose worst case is close to its typical case.

**Confidence does real work.** Jev's reported confidence tracks correctness closely, which is the
practical difference from a plain chat completion (OpenAI returns a bare label). On Banking77:

| Confidence | Accuracy | Share of calls |
| --- | --- | --- |
| <= 0.5 | 35.0% | 6.0% |
| 0.5 - 0.7 | 45.3% | 10.6% |
| 0.7 - 0.9 | 61.9% | 18.9% |
| 0.9 - 1.0 | **93.6%** | 64.5% |

That is a usable routing rule: two thirds of traffic answers at 93.6%, and the low-confidence
tail is small enough to send to a human.

**The models fail in the same places.** On Stack Exchange, of the items two models both got wrong,
they chose the *identical* wrong label 100% of the time (8/8, 10/10, 6/6 across the three pairs) -
almost entirely the `ai` / `genai` / `datascience` boundary. Cross-checking two models would not
have caught any of them.

## Method notes

- **Fairness.** Every model gets the same instruction string and the same bare list of label
  names, with no hand-written label descriptions for either side, and none of them see the train
  split. Any gap is judgement, not output formatting.
- **Intervals, not point estimates.** Accuracy is reported with a 95% Wilson interval, and
  same-row comparisons use a paired McNemar test rather than comparing two averages.
- **The older repo baselines are not comparable.** `prompt_optimisation_gepa_ace` reports Claude
  Haiku 4.5 at 79.7% / 76.1% / 71.6%, but those were scored on 201 rows drawn from the banking77
  *train* split with a different harness, so they share no rows with these runs.
