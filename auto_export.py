#!/usr/bin/env python3
"""Auto-generate PROJECT_CODE_EXPORT.md and optionally watch for file changes."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

DEFAULT_OUTPUT = "PROJECT_CODE_EXPORT.md"
IGNORED_DIRS = {
    "data",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "node_modules",
}

EXCLUDED_FILES = {
    ".env",
    ".gitignore",
}

EXCLUDED_BASENAMES = {
    "readme.md",
}

INCLUDED_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".txt",
    ".sh",
}

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".txt": "text",
    ".sh": "bash",
    ".env": "dotenv",
}


def iter_project_files(root: Path, output_file: str) -> Iterable[Tuple[str, Path]]:
    excluded_files = {output_file, "auto_export.py"}

    for dirpath, dirnames, filenames in os.walk(root):
        # Keep traversal deterministic and skip noisy/generated folders.
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for filename in sorted(filenames):
            abs_path = Path(dirpath) / filename
            rel_path = abs_path.relative_to(root).as_posix()
            if rel_path in excluded_files or filename in EXCLUDED_FILES:
                continue
            if filename.lower() in EXCLUDED_BASENAMES:
                continue
            if abs_path.suffix.lower() not in INCLUDED_SUFFIXES:
                continue
            yield rel_path, abs_path


def detect_language(path: Path) -> str:
    if path.name == ".env":
        return "dotenv"
    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "text")


def read_text_safe(path: Path) -> str:
    data = path.read_bytes()
    # Replace undecodable bytes so generation never fails on mixed encodings.
    return data.decode("utf-8", errors="replace")


def build_export(root: Path, output_file: str) -> str:
    lines = [
        "# GradEngine Code Export",
        "",
        "This document contains the full contents of code and config files only.",
        "",
    ]

    for rel_path, abs_path in iter_project_files(root, output_file):
        language = detect_language(abs_path)
        content = read_text_safe(abs_path)

        lines.append(f"## ./{rel_path}")
        lines.append("")
        lines.append(f"```{language}")
        lines.append(content)
        if content and not content.endswith("\n"):
            lines.append("")
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def write_if_changed(target: Path, content: str) -> bool:
    old_content = target.read_text(encoding="utf-8") if target.exists() else None
    if old_content == content:
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=target.parent, encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    tmp_path.replace(target)
    return True


def snapshot(root: Path, output_file: str) -> Dict[str, Tuple[int, int]]:
    state: Dict[str, Tuple[int, int]] = {}
    for rel_path, abs_path in iter_project_files(root, output_file):
        stat = abs_path.stat()
        state[rel_path] = (stat.st_mtime_ns, stat.st_size)
    return state


def run_once(root: Path, output_file: str) -> None:
    output_path = root / output_file
    export_content = build_export(root, output_file)
    changed = write_if_changed(output_path, export_content)
    if changed:
        print(f"Updated {output_file}")
    else:
        print(f"No changes: {output_file} already up to date")


def watch(root: Path, output_file: str, interval: float) -> None:
    print(f"Watching {root} (interval={interval}s) and updating {output_file}")
    previous = snapshot(root, output_file)
    run_once(root, output_file)

    try:
        while True:
            time.sleep(interval)
            current = snapshot(root, output_file)
            if current != previous:
                run_once(root, output_file)
                previous = current
    except KeyboardInterrupt:
        print("Stopped watcher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and watch PROJECT_CODE_EXPORT.md")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output markdown file path relative to project root")
    parser.add_argument("--watch", action="store_true", help="Watch for changes and auto-regenerate")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds (watch mode)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()

    if args.watch:
        watch(root, args.output, max(0.2, args.interval))
    else:
        run_once(root, args.output)


if __name__ == "__main__":
    main()
