#!/usr/bin/env python3
"""Refresh Kodkod's reviewed OpenCode catalog. Inspect the diff before shipping."""
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag in ("td", "th"):
            self.cell = ""

    def handle_data(self, text):
        if self.cell is not None:
            self.cell += text

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.row is not None:
            self.row.append((self.cell or "").strip())
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def fetch(url):
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def main():
    metadata = json.loads(fetch("https://models.dev/api.json"))
    models = []
    for service, key in [("go", "opencode-go"), ("zen", "opencode")]:
        table = Tables()
        table.feed(fetch(f"https://opencode.ai/docs/{service}/"))
        for row in table.rows:
            if len(row) != 4 or not row[2].startswith("https://opencode.ai/"):
                continue
            protocol = {
                "responses": "responses",
                "completions": "chat_completions",
                "messages": "messages",
            }.get(row[2].split("/")[-1])
            if protocol is None:
                continue
            model = metadata[key]["models"].get(row[1], {})
            models.append({
                "service": service,
                "id": row[1],
                "name": row[0],
                "protocol": protocol,
                "vision": "image" in model.get("modalities", {}).get("input", []),
                "pdf": "pdf" in model.get("modalities", {}).get("input", []),
            })
    # The live Go model list publishes Kimi K3.1 Pro before the OpenCode
    # endpoint table includes it. Its Chat Completions protocol, vision input,
    # context, and thinking options come from Kimi's first-party K3.1 docs.
    kimi_k31_pro = {
        "service": "go",
        "id": "kimi-k3.1-pro",
        "name": "Kimi K3.1 Pro",
        "protocol": "chat_completions",
        "vision": True,
        "pdf": False,
        "context_window_tokens": 1000000,
        "reasoning_efforts": ["low", "high", "max"],
        "default_reasoning_effort": "max",
    }
    for index, model in enumerate(models):
        if model["service"] == "go" and model["id"] == kimi_k31_pro["id"]:
            models[index] = {**model, **kimi_k31_pro}
            break
    else:
        models.append(kimi_k31_pro)
    if not all(any(m["service"] == service for m in models)
               for service in ("go", "zen")):
        raise ValueError("Provider endpoint tables were missing; catalog was not changed")
    path = Path(__file__).resolve().parents[1] / "kodkod-providers/src/open-code-models.json"
    path.write_text(json.dumps(models, indent=2) + "\n")


if __name__ == "__main__":
    main()
