# Software Update Verification PoC

A test-driven proof of concept for verifying operating-system update evidence
from screenshots and photographs with a vision-capable language model.

## Why this project exists

An organization collects monthly software-update confirmations from employees
through a Google Form. Each response includes an image showing the update state
of a device. The submissions are organized in Google Drive and Google Sheets,
but a person still has to open every image, interpret the operating-system
interface, and record the result manually.

This PoC was built to answer one question before investing in a production
integration:

> Can a multimodal model classify real, multilingual update screenshots
> reliably enough to automate unambiguous cases and route uncertain cases to a
> human?

The PoC answered that technical question positively. The production project
has not been implemented; this repository contains the verified AI core and a
local batch runner.

## What it does

For every local PNG or JPEG image, the pipeline:

1. validates the filesystem input and calculates its SHA-256 identity;
2. loads the image and verifies that its content has not changed;
3. sends the image, versioned prompt, and strict JSON Schema to the OpenAI
   Responses API;
4. converts SDK and API failures into an explicit failure result;
5. parses the model's JSON output;
6. validates its structure with a strict Pydantic model;
7. checks the decision against a deterministic evidence-to-decision policy;
8. prints the result and aggregate token usage.

The model returns one of three decisions:

- `accept` — the visible evidence unambiguously confirms an acceptable update
  state;
- `reject` — a required update, unfinished installation, installation error,
  restart requirement, or irrelevant image is visible;
- `manual_review` — the evidence is incomplete, ambiguous, contradictory, or
  unreadable.

The response also includes the detected platform, one or more evidence codes,
and a short explanation.

## Architecture

```text
Path
  -> Sample
  -> LoadedImage
  -> OpenAIImageAnalyzer
  -> AnalysisResponse | AnalysisFailure
  -> JSON parser
  -> VerificationResult (Pydantic)
  -> deterministic semantic policy
```

The boundaries are deliberately separate:

- `sample_io.py` owns filesystem and image-loading concerns;
- `config_builder.py` builds a reproducible prompt/schema configuration;
- `provider.py` isolates the OpenAI SDK and normalizes external failures;
- `parser.py` accepts only a JSON object;
- `models.py` protects DTO and structured-output invariants;
- `policy.py` verifies that the model's decision agrees with its evidence;
- `smoke_test.py` is a temporary local batch orchestrator.

The transport, parsing, structural validation, and semantic policy can be
adapted to other image-verification workflows. The current prompt, schema,
evidence taxonomy, and decision policy are specific to operating-system update
verification.

## Defensive behavior

The provider distinguishes:

- timeouts and connection failures;
- mapped HTTP client errors and retryable server errors;
- failed, incomplete, queued, in-progress, cancelled, and unknown response
  states;
- model refusals;
- successful structured responses.

Only a completed, non-refused response reaches the parser. A structurally valid
model answer must then pass a second, deterministic policy check. Conflicting
evidence defaults to `manual_review` rather than automatic acceptance.

## Experimental results

The selected configuration was:

- model: `gpt-4.1-mini-2025-04-14`;
- prompt: `verification_v3.txt`;
- schema: `verification_response_v1.json`;
- image detail: `low`;
- server-side response storage: disabled.

| Dataset | Images | Decision result | Exact evidence sets | Policy passes | False accepts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development | 19 | 19/19 | Not used as an unbiased metric | 18/19 | 0 |
| Evaluation | 12 | 12/12 | 11/12 | 12/12 | 0 |
| External holdout | 20 | 20/20 reviewed | 17/20 reviewed | 18/20 | 0 |

The evaluation configuration was frozen before the evaluation run. Reference
annotations were inspected only after that run completed.

The external holdout set had no pre-recorded reference annotations. Its numbers
are therefore a post-run engineering review, not a formal blinded accuracy
estimate. The datasets are small, so these results demonstrate feasibility but
must not be interpreted as a production-quality statistical guarantee.

Across the 51 images processed with the selected configuration, the runs used
146,801 input tokens and 2,325 output tokens. At the API prices used during the
experiment, the estimated total model cost was approximately USD 0.062, or
about USD 0.0012 per image. Pricing can change and should be recalculated for a
production deployment.

## Privacy

The real experimental images are intentionally excluded from this repository.
They were anonymized manually before being sent to the external API, and API
response storage was disabled.

The current code **does not implement automatic personal-data detection or
redaction**. Do not run the smoke test on images containing names, email
addresses, faces, account identifiers, device names, notifications, file paths,
or other sensitive information.

A production implementation requires a local, fail-closed privacy gateway:
images must be sanitized and rechecked before any external API request, and an
uncertain or failed privacy check must route the item to manual processing.

## Running the tests

Python 3.13 or later is required.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

The test suite uses fakes for OpenAI interactions and does not make paid API
requests. At the time of publication preparation, it contains 399 passing test
cases, including parameterized cases.

## Running the local smoke test

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` in the local copy.
2. Put only anonymized PNG or JPEG files in `smoke_test/images/`.
3. Run:

```bash
python -m software_update_verification.smoke_test smoke_test/images
```

The smoke test makes real API requests and may incur charges. The `.env` file
and the entire `smoke_test/` directory are ignored by Git.

## What is implemented

- immutable input, image, configuration, success, and failure DTOs;
- prompt and JSON Schema versioning with SHA-256 fingerprints;
- OpenAI Responses API integration with structured output;
- explicit SDK, HTTP, response-state, and refusal handling;
- strict Pydantic validation of model output;
- deterministic evidence-to-decision validation;
- sorted, non-recursive local image discovery;
- per-image failure isolation and aggregate token accounting;
- a fully fake-based automated test suite.

## What is not implemented

- Google Forms, Google Drive, or Google Sheets integration;
- automatic detection and redaction of personal data;
- employee identity processing;
- deadline tracking or reminder delivery;
- persistence, idempotent job processing, or resumable runs;
- production logging, monitoring, deployment, and access control;
- a user interface or supported command-line interface;
- production-scale evaluation and ongoing model-quality monitoring.

These omissions are intentional. The repository is a feasibility PoC, not a
production employee-compliance system.

## Repository structure

```text
prompts/    Versioned model instructions
schemas/    OpenAI-independent JSON response schema
src/        Application code
tests/      Unit and boundary tests
```

Private datasets and experiment artifacts are not included.
