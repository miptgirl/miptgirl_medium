# Jev / TypeSafe API - proof of concept

A minimal end-to-end POC for the [TypeSafe](https://docs.typesafe.ai/api) evaluation API.

TypeSafe exposes a single evaluation endpoint, `POST /v1/systemone`. You send it some `state`
(the content to judge) plus a map of **typed questions**, and get a structured answer per
question. `jev-latest` is the model that serves those requests, so a "Jev call" is a request to
`/v1/systemone` with `"model": "jev-latest"`.

The value of the typed-question design is that you never parse free text. You ask a `noul`
(yes/no), a `choice` (pick one) or a `score` (rubric rating), and get back a number or an enum
value with a confidence attached.

## Contents

- `jev_typesafe_poc.ipynb` - the API walkthrough, committed with real outputs
- `benchmark_banking77.ipynb` - Jev vs GPT on 77-intent classification
- `benchmark_stackexchange.ipynb` - Jev vs GPT on 14-topic classification (easy control)

It covers a minimal client with bearer auth and exponential backoff on `429`/`529`, one call per
question type, a batch evaluation over synthetic support tickets, confidence-based routing to
human review, and token usage tracking.

## Setup

```bash
pip install -r requirements.txt
jupyter lab jev_typesafe_poc.ipynb
```

The notebook reads the API key from `../../config.json` (outside the repo, so it is never
committed) under the `JEV_API_KEY` entry - the same pattern the other notebooks here use.

## Results from the committed run

Six synthetic tickets, one call each. Routing was correct on all six, and the urgency scores
ordered sensibly - a blocked month-end report and a lockout scored ~2.9/3, a dark-theme feature
request scored 0.5.

Two of the six fell below a 0.6 confidence threshold, which is the interesting part: the
confidence is what gives you a defensible place to draw the human-review line, rather than
trusting every verdict equally.

Total cost of the run: 6 calls, ~3.2k input tokens, 480 output tokens.
