"""Persistent offline SGLang engine with guarded Qwen3-VL media support."""
import os


class SGLangEngine:
    """Single-GPU persistent runtime (tp=dp=1), selected by CUDA_VISIBLE_DEVICES.

    Compatibility hooks verify exact upstream source hashes; they affect this
    process and its scheduler children without editing installed SGLang files.
    The pinned 0.5.6 runtime was GPU-tested with Qwen3-VL-4B-Instruct; retained
    0.5.8 guards are not a claim of validation. Do not bypass failing guards.
    A full-resolution demo input exceeded an 8192-token context in validation;
    32768 was sufficient. A context-limit status is not completed inference.
    """
    def __init__(self, model, *, context_length=32768, mem_fraction_static=0.7,
                 vlm_cache_size_mb=4096, seed=42, port=30000, attention_backend=None):
        os.environ["SGLANG_ENABLE_MM_SPLITTING"] = "1"
        os.environ["SGLANG_MM_PRECOMPUTE_HASH"] = "1"
        os.environ["SGLANG_VLM_CACHE_SIZE_MB"] = str(vlm_cache_size_mb)
        from .sglang_multimodal import build_multimodal_engine_class, guarded_source_hashes
        Engine = build_multimodal_engine_class()
        self.compatibility = guarded_source_hashes()
        kwargs = dict(model_path=model, tp_size=1, dp_size=1, dtype="bfloat16",
                      context_length=context_length, mem_fraction_static=mem_fraction_static,
                      skip_tokenizer_init=False, disable_radix_cache=False,
                      disable_overlap_schedule=True, random_seed=seed, port=port)
        if attention_backend:
            kwargs["attention_backend"] = attention_backend
        self.engine = Engine(**kwargs)
        self.context_length = context_length

    def generate(self, **kwargs):
        return self.engine.generate(**kwargs)

    def shutdown(self):
        self.engine.shutdown()
