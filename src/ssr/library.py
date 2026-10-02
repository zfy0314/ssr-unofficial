"""Reasoning library and synchronized SenseNova-style prompting.

reasoning_library.json contains six plain-text candidates and expected_actions
metadata. The metadata describes intent; it does not force a tool. --library
replaces the entries object, and descriptions in reasoning_instructions.txt are
rebuilt from those entries. Scoring uses JSON insertion order for tie-breaking;
the prompt lists descriptions by ID. tokenize_library adds thinking delimiters.
system_prompt.txt supplies the tool/answer schema; --system-prompt overrides the
complete rendered prompt. The reference checksum test protects the default text.
summary_prompts.json supplies optional page summarization; provenance.json records
source/prompt hashes. These assets are packaged with the wheel.
"""
from importlib.resources import files
import json


def asset(name):
    return files("ssr").joinpath("assets", name)


def load_library(path=None):
    from pathlib import Path
    data = json.loads((Path(path) if path else asset("reasoning_library.json")).read_text())
    entries = data.get("entries", {})
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Library must have a nonempty entries object")
    for key, entry in entries.items():
        if not isinstance(entry.get("text"), str) or not entry["text"].strip():
            raise ValueError(f"Empty reasoning candidate: {key}")
        if any(tag in entry["text"] for tag in ["<thinking>", "</thinking>", "<tool_call>", "<answer>"]):
            raise ValueError("Library entries must be plain reasoning text, without delimiters")
    return entries


def render_system_prompt(entries):
    # The synchronized reference lists descriptions by ID; scoring retains JSON
    # insertion order, which also determines the first-maximum tie break.
    descriptions = []
    for index, (_, entry) in enumerate(sorted(entries.items())):
        actions = ", ".join(entry.get("expected_actions", []))
        descriptions.append(f'[{index}] "{entry["text"].strip()}"  (expected action: {actions})')
    prefix = asset("reasoning_instructions.txt").read_text().format(
        num_candidates=len(entries), reasoning_library="\n".join(descriptions))
    return prefix + asset("system_prompt.txt").read_text()


def tokenize_library(entries, tokenizer):
    return [tokenizer.encode("<thinking>" + e["text"].strip() + "</thinking>",
                             add_special_tokens=False) for e in entries.values()]
