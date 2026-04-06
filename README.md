# Gemma 4 E2B Quantization via GitHub CI

This repo automates quantization of Gemma 4 E2B using GitHub Actions.

## Workflow
1. Trigger the workflow manually (`workflow_dispatch`).
2. The runner downloads the raw Hugging Face model (~10 GB).
3. Quantization runs twice (int4 and int8).
4. A GitHub Release is created with both builds uploaded as assets.

## Outputs
- `quantized-4bit/` → ~2–3 GB GGUF weights
- `quantized-8bit/` → ~4–5 GB GGUF weights

## Dependencies
See `requirements.txt`.

## Notes
- Int4 is smallest and most bandwidth-friendly.
- Int8 is larger but more accurate.