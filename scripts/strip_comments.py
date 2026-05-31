from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path


def strip_inline_comments(source: str) -> str:
    result_tokens: list[tokenize.TokenInfo] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            continue
        result_tokens.append(tok)
    return tokenize.untokenize(result_tokens)


def strip_module_docstring(source: str) -> str:
    tree = ast.parse(source)
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        start = tree.body[0].lineno
        end = tree.body[0].end_lineno or start
        lines = source.splitlines()
        del lines[start - 1 : end]
        while lines and not lines[0].strip():
            lines.pop(0)
        text = "\n".join(lines)
        if not text.endswith("\n"):
            text += "\n"
        return text
    return source


def strip_file(path: Path) -> None:
    source = path.read_text()
    try:
        cleaned = strip_inline_comments(source)
    except tokenize.TokenizeError as exc:
        print(f"tokenize error in {path}: {exc}", file=sys.stderr)
        return
    cleaned = strip_module_docstring(cleaned)
    try:
        compile(cleaned, str(path), "exec")
    except SyntaxError as exc:
        print(f"syntax error after cleaning {path}: {exc}", file=sys.stderr)
        return
    path.write_text(cleaned)


def main(roots: list[Path]) -> None:
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or "/.venv/" in str(path):
                continue
            strip_file(path)
            print(f"cleaned {path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    main([project_root / "app", project_root / "training",
          project_root / "tests", project_root / "scripts"])
