# Third-party notices

The bundled system/tool prompt, page-summary prompts, and image preprocessing
conventions derive from **SenseNova-MARS**, copyright (c) 2025 SenseNova,
distributed under the MIT License. Its complete notice is preserved in
[`licenses/SenseNova-MARS-MIT.txt`](licenses/SenseNova-MARS-MIT.txt).

The SSR inference controller and standalone packaging adapt the project's live
inference reference. The six library entries are synchronized without changes to
their text or expected-action metadata. Source/prompt hashes are recorded in
`src/ssr/assets/provenance.json`.

SGLang, Transformers, PyTorch, Hugging Face Hub, Pillow, Requests, Safetensors,
orjson, and Qwen model weights are dependencies with their own licenses. The
compatibility module wraps selected SGLang APIs at runtime and validates their
source hashes; it does not vendor or modify the installed SGLang source tree.
No model weights are redistributed by this repository.

## Cached demo search observations

The teaser image and five historical search results are bundled for the demo.
See `src/ssr/assets/teaser/README.md` and `search_results.json` for attribution.
Third-party search thumbnails are not relicensed under MIT.
