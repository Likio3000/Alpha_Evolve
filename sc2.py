"""
sc2.py – collect the text source of a project into one big clipboard / file
         while honouring .gitignore and avoiding duplicate output.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------- optional deps ---------- #
try:
    import pyperclip  # type: ignore
except ImportError:
    pyperclip = None
    print(
        "Note: Clipboard functionality ('pyperclip') is not available. Install it with 'uv add pyperclip'."
    )


try:
    import pathspec  # type: ignore
except ImportError:
    pathspec = None
# ----------------------------------- #

# --------- user‑tweakable knobs -------- #
ALLOWED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".md",
    ".txt",
    ".sh",
    ".bash",
    ".zsh",
    ".java",
    ".cs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".sql",
}

ALLOWED_FILENAMES = {
    "dockerfile",
    "docker-compose.yml",
    ".env.example",
    ".gitignore",
    "requirements.txt",
    "package.json",
    "composer.json",
    "pom.xml",
    "gemfile",
}

EXCLUDED_DIRS = {
    ".git",
    ".ruff_cache",
    ".svn",
    ".hg",
    "__pycache__",
    "node_modules",
    "vendor",
    "egg-info",
    "target",
    "build",
    "dist",
    "out",
    "bin",
    "obj",
    ".vscode",
    ".idea",
    ".next",
    ".venv",
    "venv",
    ".env",
    "env",
}
EXCLUDED_DIRS = {d.lower() for d in EXCLUDED_DIRS}

EXCLUDED_FILES = {
    ".env",
    "credentials.json",
    "secrets.yaml",
    "package-lock.json",
    "yarn.lock",
    "composer.lock",
}
EXCLUDED_FILES = {f.lower() for f in EXCLUDED_FILES}

MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024  # 1 MiB
DEFAULT_OUTPUT_FILE = "sc2_output.txt"
# -------------------------------------- #


def load_gitignore(project_dir: Path):
    if not pathspec:
        return None
    gi_path = project_dir / ".gitignore"
    if gi_path.exists():
        with gi_path.open(encoding="utf-8", errors="ignore") as fh:
            return pathspec.PathSpec.from_lines("gitwildmatch", fh)
    return None


def collect_project_contents(
    project_dir: Path,
    exclude_extra: set[str] | None = None,
    verbose: bool = False,
):
    gitignore_spec = load_gitignore(project_dir)
    visited: set[str] = set()
    pieces: list[str] = []

    exclude_names = EXCLUDED_FILES.union({x.lower() for x in exclude_extra or set()})

    for root, dirs, files in os.walk(project_dir, topdown=True):
        # prune unwanted dirs in‑place
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDED_DIRS]

        for filename in files:
            rel_path = os.path.relpath(os.path.join(root, filename), project_dir)
            rel_lower = rel_path.lower()

            if rel_lower in visited:
                continue
            visited.add(rel_lower)

            if filename.lower() in exclude_names:
                continue

            ext = Path(filename).suffix.lower()
            if (
                ext not in ALLOWED_EXTENSIONS
                and filename.lower() not in ALLOWED_FILENAMES
            ):
                continue

            if gitignore_spec and gitignore_spec.match_file(rel_path):
                continue

            full_path = project_dir / rel_path
            try:
                if full_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                    continue
            except OSError:
                continue

            try:
                with full_path.open("r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            except Exception:
                continue

            pieces.append(
                f"--- START FILE: {rel_path} ---\n{content}\n--- END FILE: {rel_path} ---\n"
            )

            if verbose:
                print("✓", rel_path)

    return "".join(pieces)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Copy project text files to clipboard / file"
    )
    ap.add_argument(
        "project_dir", nargs="?", default=".", help="Project directory (default: .)"
    )
    ap.add_argument(
        "-w",
        "--write",
        nargs="?",
        const=DEFAULT_OUTPUT_FILE,
        metavar="FILE",
        help=f"write collected output to FILE (default: {DEFAULT_OUTPUT_FILE})",
    )
    ap.add_argument(
        "-t", "--tests", action="store_true", help="include tests/ directory"
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true", help="print every included file"
    )
    ns = ap.parse_args()

    if not ns.tests:
        EXCLUDED_DIRS.add("tests")

    project_dir = Path(ns.project_dir).resolve()
    extra_excludes = {ns.write} if ns.write else set()

    output = collect_project_contents(
        project_dir, exclude_extra=extra_excludes, verbose=ns.verbose
    )

    if not output:
        print("No relevant files found – nothing copied.")
        sys.exit(1)

    # clipboard
    if pyperclip:
        try:
            pyperclip.copy(output)
            if ns.verbose:
                print(f"(copied {len(output):,} characters to clipboard)")
        except pyperclip.PyperclipException:
            if ns.verbose:
                print("Warning: could not access the system clipboard.")

    # optional file
    if ns.write:
        out_path = Path(ns.write).resolve()
        out_path.write_text(output, encoding="utf-8")
        print(f"Wrote {len(output):,} characters to {out_path}")

    if not ns.write and not pyperclip:
        # fall‑back: print to stdout if nowhere else
        print(output)


if __name__ == "__main__":
    main()
