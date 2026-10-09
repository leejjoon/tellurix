"""Turn a Markdown export of the Claude Doc draft into paper/index.qmd.

The Claude Doc is where the paper is drafted and commented on; this file is
the repository's copy, regenerated on request and rendered by Quarto to the
gh-pages site. The export flattens pictures to placeholders -- an uploaded
image becomes ``[image: <alt>]`` and a drawn diagram ``[embedded content:
<caption>]`` -- so ``doc_sync.json`` maps each one back to a file under
``paper/figures``. A placeholder with no mapping stops the sync rather than
silently dropping a figure.

Usage::

    python paper/sync_doc.py EXPORT [--output paper/index.qmd]

EXPORT is either the ``.md`` file or the JSON the docs connector's export
returns (the Markdown is base64 inside it).
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

PLACEHOLDER = re.compile(r"^&#91;(image|embedded content): (.*?)\\?\]\s*$")


def read_export(path: Path) -> str:
    text = path.read_text()
    if path.suffix == ".md":
        return text
    payload = json.loads(text)
    if isinstance(payload, list):
        payload = json.loads(payload[-1]["text"])
    return base64.b64decode(payload["data"]["bytes_b64"]).decode()


def figure_for(kind: str, label: str, config: dict) -> str:
    table = config["images"] if kind == "image" else config["embeds"]
    if kind == "image":
        target = table.get(label)
    else:
        # An embed is exported under its caption, which is longer than the key.
        target = next((v for k, v in table.items() if label.startswith(k)), None)
    if target is None:
        raise SystemExit(f"no figure mapped for {kind} {label!r}; add it to doc_sync.json")
    if not (HERE / target).exists():
        raise SystemExit(f"{target} is missing; render it with its script in paper/figures/")
    return f"![]({target}){{fig-alt=\"{label}\" width=100%}}"


def convert(markdown: str, config: dict) -> str:
    lines = markdown.splitlines()
    # The doc's own title and byline are replaced by the Quarto front matter.
    while lines and (lines[0].startswith("# ") or not lines[0].strip() or "· @" in lines[0]):
        lines.pop(0)

    out: list[str] = []
    skipping = False
    in_latex = False
    for line in lines:
        if line.startswith("## "):
            heading = line[3:].strip()
            skipping = any(heading.startswith(s) for s in config["exclude_sections"])
        if skipping:
            continue
        if line.strip() == "```latex":
            in_latex = True
            out.append("$$")
            continue
        if in_latex and line.strip() == "```":
            in_latex = False
            out.append("$$")
            continue
        match = PLACEHOLDER.match(line.strip())
        if match:
            kind = "image" if match.group(1) == "image" else "embed"
            out.append(figure_for(kind, match.group(2), config))
            continue
        out.append(line)
    return "\n".join(out).strip() + "\n"


def front_matter(config: dict) -> str:
    return "\n".join(
        [
            "---",
            f"title: \"{config['title']}\"",
            f"subtitle: \"{config['subtitle']}\"",
            f"author: \"{config['author']}\"",
            "date: last-modified",
            "---",
            "",
            "::: {.callout-note}",
            "This page is generated from the working draft; numbers and figures may change "
            "before submission.",
            ":::",
            "",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "index.qmd")
    args = parser.parse_args()

    config = json.loads((HERE / "doc_sync.json").read_text())
    body = convert(read_export(args.export), config)
    args.output.write_text(front_matter(config) + body)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
