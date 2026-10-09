"""Turn a Markdown export of the Claude Doc draft into the paper's pages.

The Claude Doc is where the paper is drafted and commented on; the pages under
``paper/manuscript/`` are the repository's copy, regenerated on request and
rendered by Quarto into the paper section of the gh-pages site, one page per
chapter so the sidebar and the previous/next links can navigate it. The export flattens pictures to placeholders -- an uploaded
image becomes ``[image: <alt>]`` and a drawn diagram ``[embedded content:
<caption>]`` -- so ``doc_sync.json`` maps each one back to a file under
``paper/figures``. A placeholder with no mapping stops the sync rather than
silently dropping a figure.

Usage::

    python paper/sync_doc.py EXPORT

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
    # Chapter pages sit one directory below the figures.
    return f"![](../{target}){{fig-alt=\"{label}\" width=100%}}"


def convert(markdown: str, config: dict) -> str:
    lines = markdown.splitlines()
    # The doc's own title and byline are replaced by the Quarto front matter.
    while lines and (lines[0].startswith("# ") or not lines[0].strip() or "· @" in lines[0]):
        lines.pop(0)

    # The doc carries working notes -- plans, checklists, open questions -- that
    # stay in the Claude Doc and are left out of the public page.
    out: list[str] = []
    skip_level = 0
    skip_list = False
    in_latex = False
    for line in lines:
        heading = re.match(r"^(#{2,3}) (.*)$", line)
        if heading:
            level = len(heading.group(1))
            if skip_level and level <= skip_level:
                skip_level = 0
            if not skip_level and any(heading.group(2).startswith(s) for s in config["exclude_sections"]):
                skip_level = level
            line = config["rename_headings"].get(line.strip(), line)
        if skip_level:
            continue
        if skip_list:
            if not line.strip() or line.lstrip().startswith("- ") or line.startswith("    "):
                continue
            skip_list = False
        if any(line.startswith(p) for p in config["drop_paragraph_and_list_prefixes"]):
            skip_list = True
            continue
        if any(line.lstrip().startswith(p) for p in config["drop_line_prefixes"]):
            continue
        for old, new in config["replace_text"].items():
            line = line.replace(old, new)
        for pattern in config["drop_inline_patterns"]:
            line = re.sub(pattern, "", line)
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
    # Dropped lines leave runs of blank lines behind.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


DRAFT_NOTE = (
    "::: {.callout-note}\n"
    "This paper is a working draft; numbers and figures may change before submission.\n"
    ":::\n"
)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def split_chapters(body: str) -> list[tuple[str, str]]:
    """``(heading, text)`` per level-2 section, in order."""

    parts = re.split(r"^## (.*)$", body, flags=re.M)
    return [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts), 2)]


def page(title: str, text: str) -> str:
    # Inside a chapter page its subsections become the page's own sections, so
    # the right-hand table of contents lists them.
    text = re.sub(r"^### ", "## ", text, flags=re.M)
    return f"---\ntitle: \"{title}\"\n---\n\n{text}\n"


def write_pages(body: str, config: dict, directory: Path) -> list[Path]:
    directory.mkdir(exist_ok=True)
    for old in directory.glob("*.qmd"):
        old.unlink()
    written = []
    chapters = []
    abstract = ""
    for heading, text in split_chapters(body):
        number = re.match(r"^(\d+)\.\s*(.*)$", heading)
        if number is None:
            if heading.lower() == "abstract":
                abstract = text
                continue
            raise SystemExit(f"unnumbered section {heading!r}; exclude it or number it in the doc")
        name = f"{int(number.group(1)):02d}-{slug(number.group(2))}.qmd"
        (directory / name).write_text(page(heading, text))
        written.append(directory / name)
        chapters.append((heading, name))

    contents = "\n".join(f"{i + 1}. [{h.split('. ', 1)[1]}]({n})" for i, (h, n) in enumerate(chapters))
    overview = (
        f"---\ntitle: \"{config['title']}\"\nsubtitle: \"{config['subtitle']}\"\n"
        f"author: \"{config['author']}\"\ndate: last-modified\n---\n\n"
        f"{DRAFT_NOTE}\n## Abstract\n\n{abstract}\n\n## Contents\n\n{contents}\n"
    )
    (directory / "index.qmd").write_text(overview)
    return [directory / "index.qmd", *written]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "manuscript")
    args = parser.parse_args()

    config = json.loads((HERE / "doc_sync.json").read_text())
    body = convert(read_export(args.export), config)
    for path in write_pages(body, config, args.output):
        print(f"wrote {path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
