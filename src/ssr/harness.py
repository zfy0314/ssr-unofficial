"""H: strict output validation, tool execution and multimodal history framing.

Actions contain exactly one nonempty <answer>...</answer> or a <tool_call> JSON
object with name and arguments. image_search_tool takes {}; text_search_tool
requires query; image_zoom_in_tool requires bbox_2d, label, and img_idx.
Crops use [x1,y1,x2,y2] in [0,1000] with increasing corners. img_idx is zero-based
across the original image, returned thumbnails and crops. Display placeholders
[IMAGE 1], [IMAGE 2], etc. are one-based and resolved to the same registry.
External text is escaped before ChatML insertion. Error observations never stand
in for fabricated search evidence. Image-feature cache capacity must accommodate
referenced images; increase --vlm-cache-size-mb or reduce image/context sizes if
an embedding reference cannot be retained.
"""
from __future__ import annotations

import json
import math
import re


def parse_action(text, image_count):
    text = text.strip()
    for stop in ["<|im_end|>", "<|endoftext|>"]:
        if text.endswith(stop):
            text = text[:-len(stop)].strip()
    answer = re.fullmatch(r"<answer>(.*?)</answer>", text, re.S)
    if answer and answer[1].strip() and not re.search(r"</?(?:answer|tool_call)>", answer[1]):
        return {"kind": "answer", "answer": answer[1].strip()}
    match = re.fullmatch(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    if not match:
        return {"kind": "invalid", "reason": "Expected one complete nonempty answer or tool call"}
    try:
        call = json.loads(match[1])
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise ValueError()
        name, args = call["name"], call["arguments"]
        if not isinstance(args, dict):
            raise ValueError()
        if name == "image_search_tool":
            if args:
                raise ValueError()
        elif name == "text_search_tool":
            if set(args) != {"query"} or not isinstance(args["query"], str) or not 0 < len(args["query"].strip()) <= 2000:
                raise ValueError()
        elif name == "image_zoom_in_tool":
            if set(args) != {"bbox_2d", "label", "img_idx"} or not isinstance(args["label"], str):
                raise ValueError()
            index, box = args["img_idx"], args["bbox_2d"]
            if isinstance(index, bool) or not isinstance(index, (int, float)) or not math.isfinite(index) or int(index) != index or not 0 <= index < image_count:
                raise ValueError()
            if not isinstance(box, list) or len(box) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1000 for x in box):
                raise ValueError()
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError()
        else:
            raise ValueError()
        return {"kind": "tool", **call}
    except (ValueError, TypeError, KeyError):
        return {"kind": "invalid", "reason": "Tool name or arguments fail the allowed schema"}


def literal(text):
    """External observations must not introduce chat delimiters or image markers."""
    return str(text).replace("<|", "＜|").replace("[IMAGE", "［IMAGE").replace("</tool_response>", "＜/tool_response>")


class SearchHarness:
    def __init__(self, tokenizer, media_cache, image, tools, *, mode="live"):
        from .media import MediaPayload
        self.tokenizer, self.media_cache, self.tools = tokenizer, media_cache, tools
        self.mode = mode
        self.registry = []
        self.payload = MediaPayload({})
        self.cached_assets = set()
        self.last_tool_result = None
        self.last_action = []
        self.add_image(media_cache.source(image, kind="input"))

    def add_image(self, asset):
        from .media import load_processor_tensor
        import torch
        self.registry.append(asset)
        key = asset.asset_id
        if key not in self.payload._pixels_by_asset:
            pixel, grid = load_processor_tensor(asset.manifest_record())
            self.payload._pixels_by_asset[key], self.payload._grid_by_asset[key] = pixel, grid
        self.payload.asset_ids.append(key)
        self.payload.feature_hashes.append(asset.vlm_feature_hash)
        self.payload.occurrences.append({"asset_id": key, "asset": asset.manifest_record()})
        self.payload.image_grid_thw = torch.cat([self.payload.image_grid_thw, self.payload._grid_by_asset[key]])

    def initial_context(self, system, question):
        from .media import render_media_text
        s_text = self.tokenizer.apply_chat_template([{"role": "system", "content": system}], tokenize=False)
        h_text = self.tokenizer.apply_chat_template([
            {"role": "system", "content": system},
            {"role": "user", "content": literal(question) + "\n[IMAGE 1]"}], tokenize=False, add_generation_prompt=True)
        s = self.tokenizer.encode(s_text, add_special_tokens=False)
        _, h, _ = render_media_text(h_text, self.registry, self.tokenizer, self.media_cache)
        if h[:len(s)] != s:
            raise ValueError("System prompt is not a prefix of the initial context")
        return s, h

    def prepare_media(self):
        return self.payload.cache_advance_payload(len(self.registry), set(self.payload.asset_ids)-self.cached_assets)

    def mark_media_cached(self):
        self.cached_assets.update(self.payload.asset_ids)

    def cached_media(self):
        return self.payload.cache_reference_for_count(len(self.registry))

    def parse(self, a):
        self.last_action = a
        return parse_action(self.tokenizer.decode(a, skip_special_tokens=False), len(self.registry))

    def execute(self, parsed):
        from pathlib import Path
        from .media import render_media_text
        name, args = parsed["name"], parsed["arguments"]
        delta = "continue"
        self.last_tool_result = {"name": name, "arguments": args, "status": "ok"}
        try:
            if self.mode == "none" or (self.mode == "local" and name != "image_zoom_in_tool"):
                self.last_tool_result["status"] = "tool-disabled"
                delta = "tool-disabled" if self.mode == "none" else "continue"
                available = "image_zoom_in_tool" if self.mode == "local" else "none"
                observation = (f"The requested {name} is unavailable in {self.mode} mode. "
                               "No search was executed and no evidence was returned. "
                               f"Available tools: {available}. Choose an available tool or provide a final answer "
                               "if the existing evidence is sufficient.")
            elif name == "image_zoom_in_tool":
                cropped = self.media_cache.crop(self.registry[int(args["img_idx"])], args["bbox_2d"], label=args["label"])
                self.add_image(cropped)
                observation = f"Zoomed image (registry index {len(self.registry)-1}): [IMAGE {len(self.registry)}]"
                self.last_tool_result["image"] = cropped.manifest_record()
            elif name == "image_search_tool":
                result = self.tools.image_search(self.registry[0].source_path)
                self.last_tool_result["result"] = result
                lines = ["Reverse image search results:"]
                for row in result["results"]:
                    lines += ["Title: " + literal(row["title"]), "URL: " + literal(row["url"])]
                    if row.get("image_path"):
                        self.add_image(self.media_cache.source(Path(row["image_path"]), kind="search_thumbnail"))
                        lines.append(f"Thumbnail (registry index {len(self.registry)-1}): [IMAGE {len(self.registry)}]")
                observation = "\n".join(lines) if result["results"] else "No reverse image search results."
            else:
                result = self.tools.text_search(args["query"])
                self.last_tool_result["result"] = result
                observation = "\n\n".join(literal("\n".join([row["title"], row["url"], row.get("summary", row["snippet"])])) for row in result["results"])
                observation = observation or "No text search results."
        except Exception as error:
            # Network exceptions may contain credential-bearing URLs. Log the
            # class only, while preserving a typed failure instead of an answer.
            delta = "tool-error"
            self.last_tool_result["status"] = "tool-error"
            self.last_tool_result["error_type"] = type(error).__name__
            from .tools import ToolError
            if isinstance(error, ToolError):
                self.last_tool_result["error"] = str(error)
            observation = "Tool execution failed: " + type(error).__name__
        eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        close_assistant = "" if self.last_action and self.last_action[-1] == eos else "<|im_end|>"
        tail = close_assistant + "\n<|im_start|>user\n<tool_response>\n" + observation + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
        _, o, _ = render_media_text(tail, self.registry, self.tokenizer, self.media_cache)
        self.last_tool_result["observation"] = observation
        return o, delta
