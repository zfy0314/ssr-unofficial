"""Cached image search, live SerpAPI text search and optional summaries."""
from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import ipaddress
import json
import os
from pathlib import Path
import socket
from urllib.parse import urljoin, urlsplit

import requests

from .library import asset


class ToolError(RuntimeError):
    """Safe public-facing error; never includes credential-bearing request URLs."""


def public_url(url):
    p = urlsplit(url)
    if p.scheme not in ("http", "https") or not p.hostname or p.username or p.password:
        raise ToolError("Tool result URL must be a public HTTP(S) URL")
    try:
        addresses = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except OSError:
        raise ToolError("Cannot resolve tool result host") from None
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ToolError("Non-public tool result address rejected")
    return url


def download(url, *, limit=8_000_000):
    """Validate every redirect and bound downloaded bytes; no credentials attached."""
    for _ in range(5):
        public_url(url)
        try:
            with requests.get(url, timeout=(10, 30), stream=True, allow_redirects=False,
                              headers={"User-Agent": "SSR-unofficial/0.1"}) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers.get("Location", ""))
                    continue
                if response.status_code != 200:
                    raise ToolError(f"Download HTTP {response.status_code}")
                body = bytearray()
                for chunk in response.iter_content(65536):
                    body.extend(chunk)
                    if len(body) > limit:
                        raise ToolError("Download exceeds byte limit")
                return bytes(body), response.headers.get("Content-Type", "")
        except requests.RequestException:
            raise ToolError("Tool result download failed") from None
    raise ToolError("Too many tool result redirects")


class PlainHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self.skip = max(0, self.skip-1)

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip())


class SerpTools:
    def __init__(self, cache, *, api_key=None, top_k=5, summarizer_url=None,
                 summarizer_model="qwen3-32b", image_results=None):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERP_API_KEY")
        self.top_k = top_k
        self.summarizer_url = summarizer_url
        self.summarizer_model = summarizer_model
        self.prompts = json.loads(asset("summary_prompts.json").read_text())
        self.requests_made = 0
        self.image_results = Path(image_results) if image_results else Path(str(asset("teaser/search_results.json")))

    def _request(self, method, endpoint, **kwargs):
        if not self.api_key:
            raise ToolError("Set SERPAPI_API_KEY for live searches")
        self.requests_made += 1
        try:
            response = requests.request(method, "https://serpapi.com/" + endpoint,
                                        timeout=(10, 90), **kwargs)
            if response.status_code != 200:
                raise ToolError(f"SerpAPI HTTP {response.status_code}")
            result = response.json()
        except (requests.RequestException, ValueError):
            raise ToolError("SerpAPI request failed; request details suppressed") from None
        if not isinstance(result, dict) or result.get("error"):
            raise ToolError("SerpAPI returned an error; response details suppressed")
        return json.loads(json.dumps(result).replace(self.api_key, "[REDACTED]"))

    def _cached(self, kind, identity, operation):
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.cache / f"{kind}-{digest}.json"
        if path.exists():
            result = json.loads(path.read_text())
            return result | {"cache_hit": True}
        result = operation()
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result | {"cache_hit": False}

    def image_search(self, image_path):
        """Replay recorded results only for the exact associated input image.

        Manifest schema: image_sha256, optional retrieved_at, and results with
        title, url, and image_path relative to the manifest directory. Optional
        source/thumbnail URL metadata is preserved. A re-encoded/resized query
        has a different hash and must not silently receive another image's cache.

        Live reverse search belongs here: replace the hash/manifest lookup with
        your SerpAPI image-input and search request, using _request for credential
        handling/request counts and _cached for response caching. The caller
        SearchHarness.execute expects {"results": [{"title": ..., "url": ...,
        "image_path": ...}], "cache_hit": bool}. image_path must identify a saved
        local thumbnail; the harness preprocesses and appends it to model history.
        No hosting/upload implementation is supplied. Do not invent fallback
        matches or treat the historical fixture as fresh network evidence.
        """
        manifest = json.loads(self.image_results.read_text())
        digest = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        if manifest.get("image_sha256") != digest:
            raise ToolError("No cached image search for this image; supply a matching --image-results manifest")
        rows = []
        for row in manifest["results"][:self.top_k]:
            item = dict(row)
            if item.get("image_path"):
                path = (self.image_results.parent / item["image_path"]).resolve()
                if not path.is_file():
                    raise ToolError("Cached search thumbnail is missing")
                item["image_path"] = str(path)
            rows.append(item)
        return {"results": rows, "cache_hit": True, "observation_mode": "recorded-image-search",
                "retrieved_at": manifest.get("retrieved_at"), "image_sha256": digest}

    def _summarize(self, query, content):
        """Use an optional separately served LLM; no summarizer weights bundled.

        Configure a chat API base ending in /v1 using --summarizer-url, and pass
        --summarizer-model. Optional SUMMARY_API_KEY authenticates the endpoint.
        A compatible server can be started on another available GPU with:
            python -m sglang.launch_server --model-path Qwen/Qwen3-32B
                --served-model-name qwen3-32b --host 127.0.0.1 --port 18932
                --tp-size 1 --dtype bfloat16 --context-length 32768
                --mem-fraction-static 0.8
        Join the command lines above; select a GPU using CUDA_VISIBLE_DEVICES.
        The synchronized prompt requests at most five sentences. HTML fetch or
        summary failure retains the original search snippet with a fallback flag.
        Snippet-only and page-summary observations differ and must not be mixed
        in accuracy comparisons. Tests mock network contracts; the optional live
        summary endpoint was not exercised in release validation.
        """
        headers = {}
        if os.environ.get("SUMMARY_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["SUMMARY_API_KEY"]
        messages = [{"role": "system", "content": self.prompts["SUMMARY_SYSTEM_PROMPT"]},
                    {"role": "user", "content": self.prompts["SUMMARY_USER_PROMPT"].format(
                        query=query, content=content[:30000], content_limit=30000)}]
        try:
            response = requests.post(self.summarizer_url.rstrip("/") + "/chat/completions",
                                     headers=headers, json={"model": self.summarizer_model,
                                     "messages": messages, "temperature": 0., "max_tokens": 8192,
                                     "chat_template_kwargs": {"enable_thinking": False}}, timeout=(10, 300))
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty summary")
            return text
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
            raise ToolError("Summary endpoint failed") from None

    def text_search(self, query):
        identity = dict(version=1, query=query, top_k=self.top_k, summarizer=self.summarizer_url,
                        model=self.summarizer_model, prompts=self.prompts)

        def request():
            raw = self._request("GET", "search.json", params={"engine": "google", "q": query,
                                 "num": self.top_k, "api_key": self.api_key})
            results = []
            for row in (raw.get("organic_results") or [])[:self.top_k]:
                item = {"title": row.get("title", ""), "url": row.get("link", ""), "snippet": row.get("snippet", "")}
                if self.summarizer_url and item["url"]:
                    try:
                        body, content_type = download(item["url"], limit=2_000_000)
                        if "html" not in content_type.lower():
                            raise ToolError("Search result is not HTML")
                        parser = PlainHTML()
                        parser.feed(body.decode("utf-8", errors="replace"))
                        item["summary"] = self._summarize(query, "\n".join(parser.parts))
                    except ToolError:
                        item["summary_error"] = "Using search snippet; page fetch or summary failed"
                results.append(item)
            return {"query": query, "results": results, "raw": raw,
                    "observation_mode": "page-summaries" if self.summarizer_url else "serp-snippets"}
        return self._cached("text", identity, request)
