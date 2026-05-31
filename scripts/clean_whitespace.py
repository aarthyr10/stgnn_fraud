from __future__ import annotations

import re
from pathlib import Path

HEADER_RE = re.compile(r"^\s*#\s*──")
DECORATOR_RE = re.compile(r"^\s*@\w")
DEF_OR_CLASS_RE = re.compile(r"^\s*(def|class|async\s+def)\s+")
SIGNATURE_OPEN_RE = re.compile(r"^\s*(def|class|async\s+def)\b.*[(,]\s*$")
SIGNATURE_CLOSE_RE = re.compile(r"^\s*\)\s*(->\s*[\w\[\], .]+)?\s*:\s*$")


def _indent(ln: str) -> int:
    return len(ln) - len(ln.lstrip(" "))


def _collapse_runs(lines: list[str]) -> list[str]:
    out: list[str] = []
    run = 0
    for ln in lines:
        if ln.strip() == "":
            run += 1
            continue
        if run > 0:
            allowed = 2 if _indent(ln) == 0 else 1
            for _ in range(min(run, allowed)):
                out.append("")
        run = 0
        out.append(ln.rstrip())
    while out and out[-1] == "":
        out.pop()
    out.append("")
    return out


def _join_decorator_to_def(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        if DECORATOR_RE.match(lines[i]):
            out.append(lines[i].rstrip())
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and (DECORATOR_RE.match(lines[j])
                                   or DEF_OR_CLASS_RE.match(lines[j])):
                i = j
                continue
            i += 1
            continue
        out.append(lines[i])
        i += 1
    return out


def _strip_inner_blanks(lines: list[str]) -> list[str]:
    out: list[str] = []
    in_signature = False
    just_def = False
    for ln in lines:
        stripped = ln.strip()
        if in_signature:
            if stripped == "":
                continue
            out.append(ln)
            if SIGNATURE_CLOSE_RE.match(ln) or (
                stripped.endswith(":")
                and not stripped.endswith(",")
                and "(" not in ln
            ):
                in_signature = False
                just_def = True
            continue
        if SIGNATURE_OPEN_RE.match(ln):
            out.append(ln)
            in_signature = True
            continue
        if DEF_OR_CLASS_RE.match(ln) and stripped.endswith(":"):
            out.append(ln)
            just_def = True
            continue
        if just_def and stripped == "":
            continue
        just_def = False
        out.append(ln)
    return out


def _move_headers_above_decorators(lines: list[str]) -> list[str]:
    out = list(lines)
    i = 0
    while i < len(out):
        if HEADER_RE.match(out[i]):
            j = i + 1
            while j < len(out) and out[j].strip() == "":
                j += 1
            if j < len(out) and DEF_OR_CLASS_RE.match(out[j]):
                k = i - 1
                while k >= 0 and out[k].strip() == "":
                    k -= 1
                if k >= 0 and DECORATOR_RE.match(out[k]):
                    while k > 0 and DECORATOR_RE.match(out[k - 1]):
                        k -= 1
                    header = out[i]
                    del out[i]
                    while i < len(out) and out[i].strip() == "":
                        del out[i]
                    out.insert(k, "")
                    out.insert(k, header)
                    out.insert(k, "")
                    i = k + 3
                    continue
        i += 1
    return out


def clean(path: Path) -> None:
    text = path.read_text()
    lines = text.splitlines()
    lines = _move_headers_above_decorators(lines)
    lines = _strip_inner_blanks(lines)
    lines = _join_decorator_to_def(lines)
    lines = _collapse_runs(lines)
    path.write_text("\n".join(lines))


def main(roots: list[Path]) -> None:
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            clean(path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    main([root / "app", root / "scripts", root / "training", root / "tests"])
    print("cleaned.")
