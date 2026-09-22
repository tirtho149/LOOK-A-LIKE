#!/usr/bin/env python3
"""
Frozen VLM backend — Claude Code headless (`claude -p`) with image Read.
=======================================================================
Built to mirror AgEval's ClaudeAPI contract (third_party_AgEval/inference.py):
a single call takes a text prompt + a list of images and returns a JSON dict
whose primary field is ``prediction``.  Instead of the Anthropic Messages API
we drive the local `claude -p` CLI, which reads each image path through the
Read tool.  No API key is stored in the repo.

The call is content-addressed and cached under outputs/vlm_cache/ so the whole
experiment is resume-safe (protocol §8: atomic, resume-safe JSONL/results).
"""
from __future__ import annotations
import os, re, json, time, hashlib, subprocess, tempfile
from pathlib import Path
from typing import Optional

# Reuse AgEval's JSON extraction convention verbatim (first {...} block).
def extract_json(s: str):
    m = re.search(r"\{.*\}", s or "", re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            # tolerate trailing commas / single quotes from the model
            t = m.group().replace("'", '"')
            t = re.sub(r",\s*}", "}", t); t = re.sub(r",\s*]", "]", t)
            try:
                return json.loads(t)
            except Exception:
                return None
    return None


class VLMBackend:
    def __init__(self, cfg: dict, cache_dir: Optional[Path] = None):
        m = cfg.get("model", {})
        self.cli = m.get("cli_bin", "claude")
        self.allowed_tools = m.get("allowed_tools", "Read")
        self.timeout = int(m.get("timeout_s", 300))
        self.max_retries = int(m.get("max_retries", 3))
        self.model_id = m.get("model_id", "claude-code-cli")
        self.cache_dir = Path(cache_dir or (Path(cfg["experiment_output_dir"]) / "vlm_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0

    # ---- prompt assembly -------------------------------------------------
    @staticmethod
    def _image_block(image_specs) -> str:
        """image_specs: list of (path, caption). Renders an ordered Read list."""
        if not image_specs:
            return ""
        lines = ["\nYou are given the following image(s). Read each one with the Read tool "
                 "in the order listed, then reason over them:"]
        for i, (p, cap) in enumerate(image_specs, 1):
            lines.append(f"  [{i}] {cap}: {p}")
        return "\n".join(lines) + "\n"

    def _key(self, prompt: str, image_specs) -> str:
        h = hashlib.sha256()
        h.update(prompt.encode())
        h.update(self.model_id.encode())
        for p, cap in image_specs:
            h.update(cap.encode()); h.update(str(p).encode())
            try:
                h.update(str(int(os.path.getmtime(p))).encode())
            except OSError:
                pass
        return h.hexdigest()[:24]

    # ---- core call -------------------------------------------------------
    def call(self, prompt: str, image_specs=None, expect_json: bool = True,
             tag: str = "call") -> dict:
        """Return {prediction?, raw, parsed, cache_key, latency_s, calls}."""
        image_specs = image_specs or []
        key = self._key(prompt, image_specs)
        cache_f = self.cache_dir / f"{tag}_{key}.json"
        if cache_f.exists():
            rec = json.loads(cache_f.read_text())
            rec["cached"] = True
            return rec

        full = prompt + self._image_block(image_specs)
        if expect_json:
            full += ("\n\nRespond with ONLY a single JSON object, starting with '{' "
                     "and ending with '}'. No prose, no markdown fences.")

        raw, latency, parsed = "", None, None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                proc = subprocess.run(
                    [self.cli, "-p", full, "--allowedTools", self.allowed_tools],
                    capture_output=True, text=True, timeout=self.timeout)
                raw = (proc.stdout or "").strip()
                latency = round(time.time() - t0, 2)
            except subprocess.TimeoutExpired:
                latency = round(time.time() - t0, 2); raw = ""
            parsed = extract_json(raw) if expect_json else {"text": raw}
            if (parsed is not None) or (not expect_json):
                break
            time.sleep(2 * attempt)

        self.calls += 1
        rec = {"tag": tag, "prompt_chars": len(full), "n_images": len(image_specs),
               "raw": raw, "parsed": parsed, "cache_key": key,
               "latency_s": latency, "model_id": self.model_id,
               "attempts": attempt, "cached": False}
        tmp = cache_f.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=2)); tmp.replace(cache_f)  # atomic
        return rec


if __name__ == "__main__":
    import yaml, sys
    cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
    b = VLMBackend(cfg)
    img = sys.argv[1] if len(sys.argv) > 1 else None
    specs = [(img, "QUERY IMAGE")] if img else []
    r = b.call("Identify the plant organ and any lesions. Return {\"prediction\": \"...\"}.",
               specs, tag="selftest")
    print(json.dumps({k: r[k] for k in ("parsed", "latency_s", "attempts")}, indent=2))
