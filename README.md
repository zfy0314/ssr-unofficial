# Selection-Based Structured Reasoning: Toward Efficient Multimodal Search Agents

<div align="center">
<a href="https://zfy0314.github.io/ssr-webpage/">
  <img src="https://img.shields.io/badge/-HomePage-black?logo=github" alt="homepage">
</a> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <a href="https://arxiv.org/abs/2610.01892">
  <img src="https://img.shields.io/badge/ArXiv-SSR-brown?logo=arxiv" alt="Paper">
</a>
</div>

Unofficial inference code for **Selection-Based Structured Reasoning (SSR)**.
SSR scores reusable reasoning candidates in parallel, selects one, and injects
its text into the model context before generating a tool call or answer.

## Release Timeline

- [x] Inference code, reasoning library, prompts, and tool harness.
- [ ] Training code — tentative: end of October 2026.
- [ ] 4B model weights — tentative: end of November 2026.

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

## Inference Demo

The main purpose of this demo is to showcase SSR's inference efficiency and procedure using parallel reasoning decoding.
It uses the public Qwen3-VL-4B-Instruct checkpoint to score reusable reasoning candidates in parallel, select one, and inject its text before generating a tool call or answer. 
We hope this concrete example can help the community to adopt the SSR idea to their own agent pipeline.

The example uses [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct),
which downloads automatically on first use. To download it in advance:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-4B-Instruct', local_dir='checkpoints/Qwen3-VL-4B-Instruct')"
```

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

Use `ssr-infer --help` for all options.

### Connection to the pseudocode

[`SSR.infer`](src/ssr/inference.py) implements the loop in
[Algorithm 1](https://zfy0314.github.io/ssr-webpage/#algorithm-1). The same model
scores reasoning candidates and generates the subsequent action; selection does
not require a separate classifier.

| Pseudocode step | Implementation |
| --- | --- |
| Build history `h` and candidate library `R` | `SearchHarness.initial_context` formats the image/question; `tokenize_library` prepares the full candidate texts. |
| Score `h + r_i` to obtain `S_i` | `SSR.infer` submits a batched prefill with `max_new_tokens=0`; `candidate_scores` computes length-normalized token log-likelihoods. |
| Select candidate `i_t` | `select_reasoning` chooses the first maximum at `tau=0`, or samples from `softmax(S/tau)`. |
| Inject `z_t`, then generate `a_t` | `SSR.infer` appends the complete selected reasoning to `h` and generates the action with separate sampling settings `eta`. |
| Execute the tool and update history | `SearchHarness.execute` returns an observation, which is appended before the next selection. |

Because each candidate's text is already known, its token positions can be scored
in parallel with causal attention, and candidates can be batched together. The
shared history's KV cache is reused across candidates. Only the subsequent action
is autoregressively generated.

The implementation adds an explicit shared multimodal prefill before candidate
scoring to populate image embeddings and separate prefill timing. The engine owns
the KV cache; the controller passes token sequences and media references. This
extra prefill follows the pseudocode's scoring and selection rules.

### Text and image search

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
