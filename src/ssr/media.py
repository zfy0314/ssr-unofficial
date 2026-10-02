"""Deterministic MARS image resolution, preprocessing, and token rendering."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import orjson
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor

from .hashing import atomic_write_json, semantic_hash, sha256_file


LOGGER = logging.getLogger(__name__)

MEDIA_PREPARATION_VERSION = "ssr-min65536-max8294400-v1"
VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
PLACEHOLDER_RE = re.compile(r"\[IMAGE\s+(\d+)\]")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
USER_MESSAGE_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)

IMAGE_FACTOR = 32
MIN_PIXELS = 65_536
MAX_PIXELS = 8_294_400
MIN_CROP_SIDE = 28


def _round_by_factor(value: float, factor: int) -> int:
    return round(value / factor) * factor


def _ceil_by_factor(value: float, factor: int) -> int:
    return math.ceil(value / factor) * factor


def _floor_by_factor(value: float, factor: int) -> int:
    return math.floor(value / factor) * factor


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """MARS/Qwen smart resize with a factor-32 output grid."""
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image dimensions: {width}x{height}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"image aspect ratio exceeds 200: {width}x{height}")
    resized_h = max(factor, _round_by_factor(height, factor))
    resized_w = max(factor, _round_by_factor(width, factor))
    if resized_h * resized_w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        resized_h = _floor_by_factor(int(height / beta), factor)
        resized_w = _floor_by_factor(int(width / beta), factor)
    elif resized_h * resized_w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_h = _ceil_by_factor(int(height * beta), factor)
        resized_w = _ceil_by_factor(int(width * beta), factor)
    return resized_h, resized_w


def process_mars_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    resized_h, resized_w = smart_resize(image.height, image.width)
    if image.size != (resized_w, resized_h):
        image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
    return image


def crop_normalized(image: Image.Image, bbox: Iterable[float]) -> tuple[Image.Image, list[int]]:
    values = [float(value) for value in bbox]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid zoom bbox: {values}")
    x1, y1, x2, y2 = [min(1000.0, max(0.0, value)) / 1000.0 for value in values]
    pixel_box = [int(x1 * image.width), int(y1 * image.height), int(x2 * image.width), int(y2 * image.height)]
    if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
        raise ValueError(f"empty zoom bbox after clamping: {values} -> {pixel_box}")
    cropped = image.convert("RGB").crop(tuple(pixel_box))
    if cropped.width < MIN_CROP_SIDE or cropped.height < MIN_CROP_SIDE:
        cropped = cropped.resize(
            (max(MIN_CROP_SIDE, cropped.width), max(MIN_CROP_SIDE, cropped.height)),
            Image.Resampling.LANCZOS,
        )
    return cropped, pixel_box


def _atomic_save_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".safetensors", dir=path.parent)
    os.close(descriptor)
    try:
        save_file({key: value.contiguous().cpu() for key, value in tensors.items()}, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class CachedAsset:
    asset_id: str
    kind: str
    image_path: Path
    tensor_path: Path
    source_path: Path
    source_sha256: str
    image_sha256: str
    tensor_sha256: str
    width: int
    height: int
    image_grid_thw: tuple[int, int, int]
    visual_token_count: int
    pixel_patch_count: int
    vlm_feature_hash: int
    bbox_2d: tuple[float, float, float, float] | None = None
    source_asset_id: str | None = None

    def manifest_record(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "kind": self.kind,
            "image_path": str(self.image_path.resolve()),
            "tensor_path": str(self.tensor_path.resolve()),
            "source_path": str(self.source_path.resolve()),
            "source_sha256": self.source_sha256,
            "image_sha256": self.image_sha256,
            "tensor_sha256": self.tensor_sha256,
            "width": self.width,
            "height": self.height,
            "image_grid_thw": list(self.image_grid_thw),
            "visual_token_count": self.visual_token_count,
            "pixel_patch_count": self.pixel_patch_count,
            "vlm_feature_hash": self.vlm_feature_hash,
            "bbox_2d": list(self.bbox_2d) if self.bbox_2d is not None else None,
            "source_asset_id": self.source_asset_id,
        }


class MediaObjectCache:
    """Content-addressed processed pixels and Hugging Face processor tensors."""

    def __init__(self, root: Path, model_path: str):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        model = Path(model_path).resolve() if Path(model_path).is_dir() else None
        source = str(model) if model is not None else model_path
        self.processor = AutoProcessor.from_pretrained(source, trust_remote_code=False, local_files_only=True)
        config_path = model / "preprocessor_config.json" if model is not None else None
        self.processor_fingerprint = semantic_hash({
            "class": type(self.processor).__name__,
            "preprocessor_sha256": sha256_file(config_path) if config_path and config_path.is_file() else None,
            "policy": MEDIA_PREPARATION_VERSION,
        })
        tokenizer = self.processor.tokenizer
        self.vision_start_id = int(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
        self.image_pad_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        self.vision_end_id = int(tokenizer.convert_tokens_to_ids("<|vision_end|>"))
        self.spatial_merge_size = int(getattr(self.processor.image_processor, "merge_size", 2))
        self._memory: dict[str, CachedAsset] = {}

    def _materialize(
        self,
        *,
        kind: str,
        source_path: Path,
        image: Image.Image,
        identity: dict[str, Any],
        bbox_2d: tuple[float, float, float, float] | None = None,
        source_asset_id: str | None = None,
    ) -> CachedAsset:
        source_path = source_path.resolve()
        source_sha = sha256_file(source_path)
        key = semantic_hash({
            "kind": kind,
            "source_sha256": source_sha,
            "identity": identity,
            "processor_fingerprint": self.processor_fingerprint,
            "preparation_version": MEDIA_PREPARATION_VERSION,
        })
        if key in self._memory:
            return replace(self._memory[key], source_path=source_path)
        object_dir = self.root / "objects" / key
        image_path = object_dir / "image.png"
        tensor_path = object_dir / "processor.safetensors"
        metadata_path = object_dir / "metadata.json"
        if metadata_path.is_file() and image_path.is_file() and tensor_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            feature_hash = metadata.get("vlm_feature_hash")
            if feature_hash is None:
                feature_hash = _tensor_feature_hash(load_file(tensor_path, device="cpu")["pixel_values"])
            asset = CachedAsset(
                asset_id=key,
                kind=metadata["kind"],
                image_path=image_path,
                tensor_path=tensor_path,
                source_path=source_path,
                source_sha256=metadata["source_sha256"],
                image_sha256=metadata.get("image_sha256") or sha256_file(image_path),
                tensor_sha256=metadata.get("tensor_sha256") or sha256_file(tensor_path),
                width=metadata["width"],
                height=metadata["height"],
                image_grid_thw=tuple(metadata["image_grid_thw"]),
                visual_token_count=metadata["visual_token_count"],
                pixel_patch_count=metadata["pixel_patch_count"],
                vlm_feature_hash=int(feature_hash),
                bbox_2d=tuple(metadata["bbox_2d"]) if metadata.get("bbox_2d") is not None else None,
                source_asset_id=metadata.get("source_asset_id"),
            )
            self._memory[key] = asset
            return asset

        processed = process_mars_image(image)
        encoded = self.processor.image_processor(images=[processed], do_resize=False, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(torch.float32).cpu()
        grid = encoded["image_grid_thw"].to(torch.int64).cpu()
        if grid.shape != (1, 3):
            raise ValueError(f"expected one Qwen image grid, got {tuple(grid.shape)}")
        grid_tuple = tuple(int(value) for value in grid[0].tolist())
        visual_tokens = math.prod(grid_tuple) // (self.spatial_merge_size**2)
        if pixel_values.shape[0] != math.prod(grid_tuple):
            raise ValueError("processor patch count does not match image_grid_thw")
        object_dir.mkdir(parents=True, exist_ok=True)
        temporary_png = object_dir / ".image.png.tmp"
        processed.save(temporary_png, format="PNG", optimize=False)
        os.replace(temporary_png, image_path)
        _atomic_save_safetensors(tensor_path, {"pixel_values": pixel_values, "image_grid_thw": grid})
        asset = CachedAsset(
            asset_id=key,
            kind=kind,
            image_path=image_path,
            tensor_path=tensor_path,
            source_path=source_path,
            source_sha256=source_sha,
            image_sha256=sha256_file(image_path),
            tensor_sha256=sha256_file(tensor_path),
            width=processed.width,
            height=processed.height,
            image_grid_thw=grid_tuple,
            visual_token_count=visual_tokens,
            pixel_patch_count=int(pixel_values.shape[0]),
            vlm_feature_hash=_tensor_feature_hash(pixel_values),
            bbox_2d=bbox_2d,
            source_asset_id=source_asset_id,
        )
        atomic_write_json(metadata_path, asset.manifest_record() | {
            "processor_fingerprint": self.processor_fingerprint,
            "preparation_version": MEDIA_PREPARATION_VERSION,
            "identity": identity,
        })
        self._memory[key] = asset
        return asset

    def source(self, path: Path, *, kind: str) -> CachedAsset:
        with Image.open(path) as image:
            return self._materialize(kind=kind, source_path=path, image=image.copy(), identity={"content_addressed": True})

    def crop(self, source: CachedAsset, bbox: Iterable[float], *, label: str | None = None) -> CachedAsset:
        values = tuple(float(value) for value in bbox)
        crop_source = source.source_path if source.kind != "crop" else source.image_path
        with Image.open(crop_source) as raw:
            cropped, pixel_box = crop_normalized(raw, values)
        return self._materialize(
            kind="crop",
            source_path=crop_source,
            image=cropped,
            identity={"bbox_2d": list(values), "pixel_box": pixel_box, "source_asset_id": source.asset_id},
            bbox_2d=values,
            source_asset_id=source.asset_id,
        )

    def marker_ids(self, asset: CachedAsset) -> list[int]:
        return [self.vision_start_id, *([self.image_pad_id] * asset.visual_token_count), self.vision_end_id]


def render_media_text(
    text: str,
    registry: list[CachedAsset],
    tokenizer,
    cache: MediaObjectCache,
    *,
    allow_source_marker: bool = False,
) -> tuple[str, list[int], list[dict[str, Any]]]:
    """Replace structural placeholders, tokenize once, and expand image-pad runs."""
    references: list[int] = []

    def replace(match: re.Match[str]) -> str:
        reference = int(match.group(1)) - 1
        if reference < 0 or reference >= len(registry):
            raise ValueError(f"image placeholder {match.group(0)!r} has no resolved asset")
        references.append(reference)
        return VISION_MARKER

    rendered = PLACEHOLDER_RE.sub(replace, text)
    if allow_source_marker and "<image>" in rendered:
        count = rendered.count("<image>")
        if count != 1:
            raise ValueError(f"expected one source <image> marker, found {count}")
        references.insert(0, 0)
        rendered = rendered.replace("<image>", VISION_MARKER)
    marker_ids = tokenizer.encode(VISION_MARKER, add_special_tokens=False)
    expected_marker = [cache.vision_start_id, cache.image_pad_id, cache.vision_end_id]
    if marker_ids != expected_marker:
        raise ValueError(f"unexpected tokenizer vision marker IDs: {marker_ids}")
    compact_ids = tokenizer.encode(rendered, add_special_tokens=False)
    expanded: list[int] = []
    spans: list[dict[str, Any]] = []
    reference_index = 0
    cursor = 0
    while cursor < len(compact_ids):
        if compact_ids[cursor : cursor + 3] == expected_marker:
            if reference_index >= len(references):
                raise ValueError("tokenized vision marker has no media reference")
            registry_index = references[reference_index]
            asset = registry[registry_index]
            start = len(expanded)
            ids = cache.marker_ids(asset)
            expanded.extend(ids)
            spans.append({
                "registry_index": registry_index,
                "placeholder_number": registry_index + 1,
                "asset_id": asset.asset_id,
                "token_start": start,
                "token_end": len(expanded),
                "visual_token_count": asset.visual_token_count,
            })
            reference_index += 1
            cursor += 3
        else:
            expanded.append(compact_ids[cursor])
            cursor += 1
    if reference_index != len(references):
        raise ValueError("media reference count differs from tokenized marker count")
    return rendered, expanded, spans


def load_processor_tensor(asset: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = load_file(asset["tensor_path"], device="cpu")
    return tensors["pixel_values"], tensors["image_grid_thw"]


def _tensor_feature_hash(tensor: torch.Tensor) -> int:
    """Match SGLang 0.5.8's first-eight-byte SHA-256 tensor cache key."""
    value = tensor.detach().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.float()
    digest = hashlib.sha256(memoryview(value.cpu().numpy()).tobytes()).digest()[:8]
    return int.from_bytes(digest, byteorder="big", signed=False)


class MediaPayload:
    """Resident occurrence-ordered tensors; payload construction is outside engine timers."""

    def __init__(self, trajectory_record: dict[str, Any]):
        occurrences = trajectory_record.get("media_occurrences") or []
        self.occurrences = occurrences
        grids: list[torch.Tensor] = []
        self.asset_ids: list[str] = []
        self.feature_hashes: list[int] = []
        self._pixels_by_asset: dict[str, torch.Tensor] = {}
        self._grid_by_asset: dict[str, torch.Tensor] = {}
        for occurrence in occurrences:
            asset_id = occurrence["asset_id"]
            if asset_id not in self._pixels_by_asset:
                pixel, grid = load_processor_tensor(occurrence["asset"])
                self._pixels_by_asset[asset_id] = pixel
                self._grid_by_asset[asset_id] = grid
            grid = self._grid_by_asset[asset_id]
            grids.append(grid)
            self.asset_ids.append(asset_id)
            self.feature_hashes.append(int(occurrence["asset"]["vlm_feature_hash"]))
        self.image_grid_thw = torch.cat(grids, dim=0) if grids else torch.empty((0, 3), dtype=torch.int64)

    def cache_advance_payload(
        self, count: int, uncached_asset_ids: Iterable[str]
    ) -> dict[str, Any] | None:
        """Send references for the full prefix and pixels only for cache misses."""
        if count == 0:
            return None
        if count < 0 or count > len(self.occurrences):
            raise ValueError(f"invalid media prefix count {count}/{len(self.occurrences)}")
        prefix_assets = set(self.asset_ids[:count])
        requested = set(uncached_asset_ids)
        if not requested <= prefix_assets:
            raise ValueError("uncached media assets are not all present in the requested prefix")
        missing_asset_ids = list(dict.fromkeys(
            asset_id for asset_id in self.asset_ids[:count] if asset_id in requested
        ))
        pixels = [self._pixels_by_asset[asset_id] for asset_id in missing_asset_ids]
        missing_grids = [self._grid_by_asset[asset_id] for asset_id in missing_asset_ids]
        feature_hash_by_asset = {
            asset_id: feature_hash
            for asset_id, feature_hash in zip(self.asset_ids, self.feature_hashes, strict=True)
        }
        return {
            "format": "ssr_hybrid_reference",
            "image_grid_thw": self.image_grid_thw[:count],
            "feature_hashes": self.feature_hashes[:count],
            "missing_asset_ids": missing_asset_ids,
            "missing_feature_hashes": [feature_hash_by_asset[asset_id] for asset_id in missing_asset_ids],
            "missing_pixel_values": (
                torch.cat(pixels, dim=0) if pixels else torch.empty((0, 1536), dtype=torch.float32)
            ),
            "missing_image_grid_thw": (
                torch.cat(missing_grids, dim=0) if missing_grids else torch.empty((0, 3), dtype=torch.int64)
            ),
        }

    def cache_reference_for_count(self, count: int) -> dict[str, Any] | None:
        """Compact payload for a prefix whose image embeddings were just cached."""
        if count == 0:
            return None
        if count < 0 or count > len(self.occurrences):
            raise ValueError(f"invalid media prefix count {count}/{len(self.occurrences)}")
        return {
            "format": "ssr_cache_reference",
            "image_grid_thw": self.image_grid_thw[:count],
            "feature_hashes": self.feature_hashes[:count],
        }
