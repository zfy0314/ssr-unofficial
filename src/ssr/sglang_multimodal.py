"""Version-gated SGLang 0.5.6/0.5.8 compatibility for prepared Qwen3-VL inputs.

The installed package is never edited.  The tokenizer-side patch preserves grid
metadata for ``processor_output`` dictionaries.  The scheduler-side patch makes
the VLM embedding cache image-addressable instead of keying a growing image set
as a single object.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SOURCE_SHA256_BY_VERSION = {
    "0.5.6": {
        "processor": "ea2c2f9dfdf2b58b1021fb542ec4c0dde86019f633e6a83ea79e9cea0fc21030",
        "qwen_processor": "e917ddd9f64af077b76d8bb335381da3db3791deed055967d32e1e245a661675",
        "scheduler_cache": "2f53296d61ae0ede715ed60932248560041291d535f683c731ccc09ab929ef7a",
        "split": None,
    },
    "0.5.8": {
        "processor": "431d6aa4f515c83158a475519781b0c4c4958982d0ed39da449b9346fd3f3bde",
        "qwen_processor": "29d7a2147e2bf06342d6bbdb1798aa6678dcc986cf732372f90215034860728a",
        "scheduler_cache": "11132c22f26b0e46b57cd63b31345311cc834b6f6a4506e793c2e5feee708650",
        "split": "5b88101557c73dcd53edc83dbdbd19b70d7e2afea22673d40b728b539ebef847",
    },
}


def _source_hash(value: Any) -> str:
    return hashlib.sha256(inspect.getsource(value).encode()).hexdigest()


def _require_supported() -> tuple[str, str]:
    version = importlib.metadata.version("sglang")
    base = ".".join(version.split(".")[:3])
    if base not in SOURCE_SHA256_BY_VERSION:
        raise RuntimeError(
            "SSR multimodal mode requires guarded SGLang 0.5.6 or 0.5.8, "
            f"found {version}"
        )
    return version, base


def guarded_source_hashes() -> dict[str, Any]:
    version, base = _require_supported()
    return {"sglang": version, "sglang_base": base, **SOURCE_SHA256_BY_VERSION[base]}


def _split_image_items_without_tensor_copies(items):
    """SGLang's image split semantics without deepcopying the full bundle N times."""
    import torch

    def length(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return len(value)
        return value.shape[0] if hasattr(value, "shape") and len(value.shape) > 0 else None

    output = []
    for item in items:
        if not item.is_image() or item.offsets is None or len(item.offsets) <= 1:
            output.append(item)
            continue
        grids = item.model_specific_data.get("image_grid_thw")
        if grids is None or length(grids) != len(item.offsets):
            output.append(item)
            continue
        patch_counts = [int(torch.prod(torch.as_tensor(grid, dtype=torch.long)).item()) for grid in grids]
        total = sum(patch_counts)
        feature = item.feature if item.feature is not None else item.precomputed_embeddings
        if length(feature) != total:
            output.append(item)
            continue
        cursor = 0
        for index, patch_count in enumerate(patch_counts):
            end = cursor + patch_count
            model_data = {}
            for key, value in item.model_specific_data.items():
                value_length = length(value)
                if value_length == len(item.offsets):
                    model_data[key] = value[index : index + 1]
                elif value_length == total:
                    model_data[key] = value[cursor:end]
                else:
                    model_data[key] = value
            item_kwargs = dict(
                modality=item.modality,
                hash=None,
                pad_value=None,
                offsets=[item.offsets[index]],
                feature=item.feature[cursor:end] if item.feature is not None else None,
                precomputed_embeddings=(
                    item.precomputed_embeddings[cursor:end]
                    if item.precomputed_embeddings is not None else None
                ),
                model_specific_data=model_data,
            )
            if "format" in {field.name for field in dataclasses.fields(type(item))}:
                item_kwargs["format"] = item.format
            output.append(type(item)(**item_kwargs))
            cursor = end
    return output


def apply_tokenizer_processor_patch() -> dict[str, str]:
    """Make formatted processor grids visible to Qwen's MRoPE calculation."""
    version, base = _require_supported()
    source_hashes = SOURCE_SHA256_BY_VERSION[base]
    from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

    current = BaseMultimodalProcessor.process_and_combine_mm_data
    if getattr(current, "_ssr_patch", False):
        return {"sglang": version, "processor_source_sha256": source_hashes["processor"]}
    actual = _source_hash(current)
    if actual != source_hashes["processor"]:
        raise RuntimeError(f"SGLang processor source drift: expected {source_hashes['processor']}, got {actual}")
    original = current

    def patched(self, base_output, mm_tokens, **kwargs):
        items, input_ids, processor_result = original(self, base_output, mm_tokens, **kwargs)
        if processor_result is None:
            formatted = [
                value
                for value in getattr(base_output, "images", [])
                if isinstance(value, dict) and value.get("format") == "processor_output"
            ]
            if len(formatted) == 1:
                processor_result = SimpleNamespace(**formatted[0])
        return items, input_ids, processor_result

    patched._ssr_patch = True
    patched._ssr_original = original
    BaseMultimodalProcessor.process_and_combine_mm_data = patched

    import torch
    from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

    qwen_current = QwenVLImageProcessor.process_mm_data_async
    if not getattr(qwen_current, "_ssr_patch", False):
        qwen_actual = _source_hash(qwen_current)
        if qwen_actual != source_hashes["qwen_processor"]:
            raise RuntimeError(
                f"SGLang Qwen processor source drift: expected {source_hashes['qwen_processor']}, got {qwen_actual}"
            )
        qwen_original = qwen_current

        async def qwen_patched(self, image_data, input_text, request_obj, *args, **kwargs):
            is_hybrid = (
                isinstance(input_text, list)
                and image_data
                and isinstance(image_data[0], dict)
                and image_data[0].get("format") == "ssr_hybrid_reference"
            )
            is_reference = (
                isinstance(input_text, list)
                and image_data
                and isinstance(image_data[0], dict)
                and image_data[0].get("format") == "ssr_cache_reference"
            )
            if is_reference or is_hybrid:
                from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

                exact_ids = [int(value) for value in input_text]
                exact_tensor = torch.tensor(exact_ids, dtype=torch.long)
                reference = image_data[0]
                image_grid_thw = torch.as_tensor(reference["image_grid_thw"], dtype=torch.int64)
                hashes = [int(value) for value in reference["feature_hashes"]]
                offsets = self.get_mm_items_offset(
                    input_ids=exact_tensor,
                    mm_token_id=self.mm_tokens.image_token_id,
                )
                if len(offsets) != len(hashes) or len(hashes) != image_grid_thw.shape[0]:
                    raise RuntimeError("compact VLM references, offsets, and grids are not aligned")
                features_by_hash = {}
                if is_hybrid:
                    missing_hashes = [int(value) for value in reference["missing_feature_hashes"]]
                    missing_grids = torch.as_tensor(reference["missing_image_grid_thw"], dtype=torch.int64)
                    missing_pixels = torch.as_tensor(reference["missing_pixel_values"])
                    if len(missing_hashes) != missing_grids.shape[0] or len(set(missing_hashes)) != len(missing_hashes):
                        raise RuntimeError("hybrid VLM misses, hashes, and grids are not aligned")
                    cursor = 0
                    for feature_hash, grid in zip(missing_hashes, missing_grids, strict=True):
                        end = cursor + int(torch.prod(grid).item())
                        features_by_hash[feature_hash] = missing_pixels[cursor:end]
                        cursor = end
                    if cursor != missing_pixels.shape[0]:
                        raise RuntimeError("hybrid VLM pixel tensor does not match missing image grids")
                mm_items = []
                item_fields = {field.name for field in dataclasses.fields(MultimodalDataItem)}
                for index, (feature_hash, offset) in enumerate(zip(hashes, offsets, strict=True)):
                    item_kwargs = dict(
                        modality=Modality.IMAGE,
                        hash=feature_hash,
                        pad_value=feature_hash % (1 << 30),
                        offsets=[offset],
                        feature=features_by_hash.get(feature_hash),
                        model_specific_data={"image_grid_thw": image_grid_thw[index : index + 1]},
                    )
                    if "format" in item_fields:
                        from sglang.srt.managers.schedule_batch import MultimodalInputFormat

                        item_kwargs["format"] = MultimodalInputFormat.PROCESSOR_OUTPUT
                    mm_items.append(MultimodalDataItem(**item_kwargs))
                mrope_positions, mrope_delta = MRotaryEmbedding.get_rope_index(
                    spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
                    image_token_id=self.mm_tokens.image_token_id,
                    video_token_id=self.mm_tokens.video_token_id,
                    vision_start_token_id=self.vision_start_token_id,
                    model_type=self.model_type,
                    tokens_per_second=getattr(self.hf_config.vision_config, "tokens_per_second", None),
                    input_ids=exact_tensor.unsqueeze(0),
                    image_grid_thw=image_grid_thw,
                )
                return {
                    "input_ids": exact_ids,
                    "mm_items": mm_items,
                    "im_start_id": self.vision_start_token_id,
                    "im_end_id": self.vision_end_token_id,
                    "im_token_id": self.mm_tokens.image_token_id,
                    "video_token_id": self.mm_tokens.video_token_id,
                    "audio_token_id": self.audio_token_id,
                    "mrope_positions": mrope_positions.squeeze(1),
                    "mrope_position_delta": mrope_delta,
                }

            result = await qwen_original(self, image_data, input_text, request_obj, *args, **kwargs)
            is_prepared = (
                isinstance(input_text, list)
                and image_data
                and isinstance(image_data[0], dict)
                and image_data[0].get("format") == "processor_output"
            )
            if not is_prepared:
                return result
            exact_ids = [int(value) for value in input_text]
            exact_tensor = torch.tensor(exact_ids, dtype=torch.long)
            for item in result["mm_items"]:
                item.offsets = self.get_mm_items_offset(
                    input_ids=exact_tensor,
                    mm_token_id=self.mm_tokens.image_token_id,
                )
            from sglang.srt.managers import mm_utils

            split_source = source_hashes["split"]
            if split_source is not None:
                if _source_hash(mm_utils.get_new_expanded_mm_items) != split_source:
                    raise RuntimeError("SGLang multimodal split source drift in tokenizer process")
            split_items = _split_image_items_without_tensor_copies(result["mm_items"])
            if len(split_items) != image_data[0]["image_grid_thw"].shape[0]:
                raise RuntimeError("processor output did not split into one item per image")
            for item in split_items:
                item.hash = None
                item.pad_value = None
                item.set_pad_value()
            result["mm_items"] = split_items
            image_grid_thw = image_data[0]["image_grid_thw"]
            mrope_positions, mrope_delta = MRotaryEmbedding.get_rope_index(
                spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
                image_token_id=self.mm_tokens.image_token_id,
                video_token_id=self.mm_tokens.video_token_id,
                vision_start_token_id=self.vision_start_token_id,
                model_type=self.model_type,
                tokens_per_second=getattr(self.hf_config.vision_config, "tokens_per_second", None),
                input_ids=exact_tensor.unsqueeze(0),
                image_grid_thw=image_grid_thw,
            )
            result["input_ids"] = exact_ids
            result["mrope_positions"] = mrope_positions.squeeze(1)
            result["mrope_position_delta"] = mrope_delta
            return result

        qwen_patched._ssr_patch = True
        qwen_patched._ssr_original = qwen_original
        QwenVLImageProcessor.process_mm_data_async = qwen_patched
    return {
        "sglang": version,
        "processor_source_sha256": actual,
        "qwen_processor_source_sha256": source_hashes["qwen_processor"],
    }


def apply_scheduler_cache_patch() -> None:
    """Install the per-image cache implementation inside the spawned scheduler."""
    _, base = _require_supported()
    source_hashes = SOURCE_SHA256_BY_VERSION[base]
    import torch
    from sglang.srt.managers import mm_utils
    from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache

    EmbeddingResult = None
    if base == "0.5.8":
        from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult as SGLangEmbeddingResult

        EmbeddingResult = SGLangEmbeddingResult

    current = mm_utils._get_chunked_prefill_embedding
    if getattr(current, "_ssr_patch", False):
        return
    actual = _source_hash(current)
    if actual != source_hashes["scheduler_cache"]:
        raise RuntimeError(
            f"SGLang VLM cache source drift: expected {source_hashes['scheduler_cache']}, got {actual}"
        )
    stats = {"lookups": 0, "hits": 0, "misses": 0, "encoded_images": 0, "encoded_batches": 0}

    split_source = source_hashes["split"]
    if split_source is not None:
        split_current = mm_utils.get_new_expanded_mm_items
        split_actual = _source_hash(split_current)
        if split_actual != split_source:
            raise RuntimeError(
                f"SGLang multimodal split source drift: expected {split_source}, got {split_actual}"
            )

        def split_patched(original_mm_items):
            items = _split_image_items_without_tensor_copies(original_mm_items)
            for item in items:
                if item.hash is None:
                    # Recompute both fields from the per-image tensor so the
                    # radix key remains stable as later images are appended.
                    item.pad_value = None
                    item.set_pad_value()
            return items

        split_patched._ssr_patch = True
        split_patched._ssr_original = split_current
        mm_utils.get_new_expanded_mm_items = split_patched

    def per_image_embedding(
        data_embedding_func,
        embedding_items,
        items_size,
        prefix_length,
        extend_length,
        items_offset_list,
    ):
        embedding_chunks = []
        max_iterations = min(len(items_size) - 1, len(prefix_length))
        for request_index in range(max_iterations):
            start = items_size[request_index]
            end = items_size[request_index + 1]
            if start == end:
                continue
            request_items = embedding_items[start:end]
            request_offsets = items_offset_list[request_index]
            if request_offsets is None:
                raise RuntimeError("multimodal request has no offsets")
            if len(request_items) != len(request_offsets):
                raise RuntimeError("per-image split items and offsets are not aligned")
            prefix = prefix_length[request_index]
            extension = extend_length[request_index] if request_index < len(extend_length) else 0
            if extension <= 0 or all(offset_end < prefix for _, offset_end in request_offsets):
                continue
            selected_pairs = [
                (item, offset)
                for item, offset in zip(request_items, request_offsets, strict=True)
                if offset[1] >= prefix and offset[0] < prefix + extension
            ]
            if not selected_pairs:
                continue

            cached: dict[int, Any] = {}
            missing: dict[int, Any] = {}
            for item, _ in selected_pairs:
                if item.hash is None:
                    item.pad_value = None
                    item.set_pad_value()
                key = int(item.hash)
                if key in cached or key in missing:
                    continue
                stats["lookups"] += 1
                result = mm_utils.embedding_cache.get([key])
                if result is None:
                    stats["misses"] += 1
                    missing[key] = item
                else:
                    stats["hits"] += 1
                    cached[key] = result
            if missing:
                missing_items = list(missing.values())
                if any(item.feature is None and item.precomputed_embeddings is None for item in missing_items):
                    raise RuntimeError(
                        "compact VLM cache reference missed; cache-advance must populate every referenced image"
                    )
                encoded = data_embedding_func(missing_items)
                if not isinstance(encoded, torch.Tensor):
                    raise RuntimeError("per-image SSR cache supports tensor Qwen embeddings only")
                token_lengths = [item.offsets[0][1] - item.offsets[0][0] + 1 for item in missing_items]
                if sum(token_lengths) != encoded.shape[0]:
                    raise RuntimeError(
                        f"ViT output length {encoded.shape[0]} != split visual tokens {sum(token_lengths)}"
                    )
                stats["encoded_batches"] += 1
                stats["encoded_images"] += len(missing_items)
                cursor = 0
                for key, length in zip(missing, token_lengths, strict=True):
                    embedding = encoded[cursor : cursor + length]
                    result = EmbeddingResult(embedding=embedding) if EmbeddingResult is not None else embedding
                    cache_key = MultiModalStaticCache.combine_hashes([key])
                    if not mm_utils.embedding_cache.set(cache_key, result):
                        raise RuntimeError("VLM cache cannot hold one image embedding; increase --vlm-cache-size-mb")
                    cached[key] = result
                    cursor += length

            for item, offset in selected_pairs:
                result = cached.get(int(item.hash))
                if result is None:
                    result = mm_utils.embedding_cache.get([int(item.hash)])
                if result is None:
                    raise RuntimeError("per-image embedding was evicted before request assembly")
                embedding = result.embedding if EmbeddingResult is not None else result
                chunk, _, _ = mm_utils.get_embedding_chunk(
                    embedding=embedding,
                    extend_prefix_len=prefix,
                    extend_seq_len=extension,
                    items_offset=[offset],
                )
                if chunk.numel():
                    embedding_chunks.append(chunk)
        return torch.cat(embedding_chunks, dim=0) if embedding_chunks else None

    if base == "0.5.6":
        def patched(
            data_embedding_func,
            embedding_items,
            items_size,
            prefix_length,
            extend_length,
            items_offset_list,
        ):
            return per_image_embedding(
                data_embedding_func,
                embedding_items,
                items_size,
                prefix_length,
                extend_length,
                items_offset_list,
            )
    else:
        def patched(
            data_embedding_func,
            embedding_items,
            items_size,
            prefix_length,
            extend_length,
            items_offset_list,
            input_ids,
        ):
            embedding = per_image_embedding(
                data_embedding_func,
                embedding_items,
                items_size,
                prefix_length,
                extend_length,
                items_offset_list,
            )
            return embedding, input_ids

    patched._ssr_patch = True
    patched._ssr_original = current
    mm_utils._get_chunked_prefill_embedding = patched
    mm_utils._ssr_vlm_stats = stats

    from sglang.srt.managers.scheduler import Scheduler

    def dump_snapshot(self, path: str):
        cache = mm_utils.embedding_cache
        payload = {
            "stats": dict(mm_utils._ssr_vlm_stats),
            "entries": len(cache.mm_cache),
            "current_size_bytes": int(cache.current_size),
            "max_size_bytes": int(cache.max_size),
            "keys": [str(key) for key in cache.mm_cache],
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, destination)

    Scheduler.ssr_dump_vlm_cache = dump_snapshot


def run_patched_scheduler_process(*args, **kwargs):
    """Picklable Engine process target used by the project-owned Engine subclass."""
    apply_scheduler_cache_patch()
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


def build_multimodal_engine_class():
    apply_tokenizer_processor_patch()
    _, base = _require_supported()
    from sglang.srt.entrypoints import engine as engine_module

    if base == "0.5.6":
        # 0.5.6 captures this module global as the multiprocessing target;
        # it does not yet expose Engine.run_scheduler_process_func.
        engine_module.run_scheduler_process = run_patched_scheduler_process
        return engine_module.Engine

    Engine = engine_module.Engine

    class SSRMultimodalEngine(Engine):
        run_scheduler_process_func = staticmethod(run_patched_scheduler_process)

    return SSRMultimodalEngine
