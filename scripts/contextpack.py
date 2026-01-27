#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(".").resolve()
OUT = ROOT / ".planning" / "CONTEXTPACK.md"

IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "node_modules",
}

KEY_FILES = [
    "CLAUDE.md",
    "PROJECT.md",
    "REQUIREMENTS.md",
    "ROADMAP.md",
    "STATE.md",
    "pyproject.toml",
    ".pre-commit-config.yaml",
]

def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as e:
        return f"(failed: {' '.join(cmd)}): {e}"

def tree(max_depth: int = 3) -> str:
    lines: list[str] = []

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for p in sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            if p.name in IGNORE_DIRS:
                continue
            rel = p.relative_to(ROOT)
            indent = "  " * depth
            if p.is_dir():
                lines.append(f"{indent}📁 {rel}/")
                walk(p, depth + 1)
            else:
                lines.append(f"{indent}📄 {rel}")

    walk(ROOT, 0)
    return "\n".join(lines)

def head(path: Path, n: int = 160) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[:n])
    except Exception as e:
        return f"(failed reading {path}): {e}"

def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    parts.append("# Context Pack")
    parts.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")

    parts.append("## Repo tree\n```text\n" + tree() + "\n```")
    parts.append("## Git status\n```text\n" + run(["git", "status"]) + "\n```")
    parts.append("## Recent commits\n```text\n" + run(["git", "--no-pager", "log", "-10", "--oneline"]) + "\n```")

    parts.append("## Key files (head)")
    for f in KEY_FILES:
        p = ROOT / f
        if p.exists():
            parts.append(f"### {f}\n```text\n{head(p)}\n```")

    OUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"✅ Wrote {OUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
