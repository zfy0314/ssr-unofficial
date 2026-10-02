# Selection-Based Structured Reasoning: Toward Efficient Multimodal Search Agents

Unofficial inference code for **Selection-Based Structured Reasoning (SSR)**.
SSR scores reusable reasoning candidates in parallel, selects one, and injects
its text into the model context before generating a tool call or answer.

[Project webpage](https://zfy0314.github.io/ssr-webpage/)

## Timeline

- [x] Inference code, reasoning library, prompts, and tool harness.
- [ ] Training code — tentative: end of October 2026.
- [ ] Sample 4B checkpoint — tentative: end of November 2026.

## Installation

Requires Linux, an NVIDIA GPU, and Python 3.12. The inference dependencies pin
SGLang 0.5.6, PyTorch 2.9.1, and Transformers 4.57.1.

```bash
git clone https://github.com/zfy0314/ssr-unofficial.git
cd ssr-unofficial
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[inference]'
```

The example uses [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct),
which downloads automatically on first use. To download it in advance:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-4B-Instruct', local_dir='checkpoints/Qwen3-VL-4B-Instruct')"
```

## Quick start: exhibition example

The bundled example asks:

> What is the title of the exhibition that will be hosted at this location on how
> people are experiencing profound disruption?

```bash
export SERPAPI_API_KEY='your-serpapi-key'
CUDA_VISIBLE_DEVICES=0 ssr-infer \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --input-jsonl examples/teaser-exhibition.jsonl \
  --tools live --search-results 5 \
  --max-turns 4 --max-new-tokens 1024 \
  --profiling --output outputs/exhibition
```

| Flag | Purpose |
| --- | --- |
| `--model` | Hugging Face model ID or checkpoint directory. |
| `--input-jsonl` | Image/question pairs, one JSON object per line. |
| `--image`, `--question` | Single-input alternative to JSONL. |
| `--tools` | `live` (default), crop-only `local`, or emit-only `none`. |
| `--max-turns`, `--max-new-tokens` | Turn limit and action-token limit per turn. |
| `--alpha`, `--tau` | Score normalization and candidate-selection temperature. |
| `--temperature`, `--top-p`, `--top-k` | Action-token sampling controls. |
| `--num-samples`, `--seed` | Repeat count and random seed. |
| `--search-results` | Maximum search results returned. |
| `--summarizer-url`, `--summarizer-model` | Optional page-summary LLM endpoint and model. |
| `--image-results`, `--cache` | Image-result manifest and cache directory. |
| `--library`, `--system-prompt` | Override reasoning candidates or prompt. |
| `--context-length`, `--mem-fraction-static`, `--vlm-cache-size-mb` | Context and GPU/cache limits. |
| `--port`, `--attention-backend` | Engine port and optional attention backend. |
| `--offline`, `--dry-run` | Use local weights only, or validate without inference. |
| `--profiling`, `--output` | Print trajectory totals; choose a new output directory. |

A final answer can stop before the turn limit. Full traces and per-turn/trajectory
statistics are saved in the output directory. Use `ssr-infer --help` for all options.

## Text and image search

**Text search** uses SerpAPI and returns titles, URLs, and snippets. Export
`SERPAPI_API_KEY` (`SERP_API_KEY` also works). An optional chat-completions endpoint
specified by `--summarizer-url` and `--summarizer-model` summarizes fetched pages;
otherwise the snippets are returned. No summarizer weights are bundled.

**Image search currently replays cached results.** The
[manifest](src/ssr/assets/teaser/search_results.json) contains the input image's
SHA-256, retrieval date, titles, URLs, and thumbnail filenames. Thumbnails are
bundled beside it. The input hash must match; `--image-results` selects another
recorded manifest. Text-search responses are also cached to avoid repeated calls.

To add live SerpAPI reverse-image search, replace `SerpTools.image_search` in
[`src/ssr/tools.py`](src/ssr/tools.py). Its docstring describes the result format
and the existing request/cache helpers to reuse.

## Attribution and license

The code is [MIT licensed](LICENSE). SenseNova-MARS attribution is retained for
the bundled prompts and preprocessing; see [third-party notices](THIRD_PARTY_NOTICES.md).
The runtime uses SGLang, Transformers, and Qwen3-VL. Model weights and third-party
search thumbnails retain their own licenses and rights.
