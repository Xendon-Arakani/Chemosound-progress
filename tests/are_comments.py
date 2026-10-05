import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAIL_TEXT = "Comments and documentation outside explicitly required data saving are not allowed."
NAME_LIMIT = 20
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "dist", "build"}
DATA_ROOTS = {".chemosound", "local-data", "history"}
DOC_TYPES = {".md", ".rst", ".adoc", ".tex", ".doc", ".docx"}
TEXT_CODE = {".toml", ".ps1", ".sql", ".sh", ".js", ".ts", ".css", ".c", ".cpp", ".h", ".hpp", ".html", ".htm"}
RAW_ROOTS = (
    ".chemosound/tools/flac-1.5.0/",
    "local-data/Chemosound-data/hitran2024/source-software/",
    "local-data/Chemosound-data/hitran2024/source-metadata/",
)


def owned_files():
    files = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts) or not path.is_file():
            continue
        if any(rel.as_posix().startswith(prefix) for prefix in RAW_ROOTS):
            continue
        files.append(path)
    return files


def name_bindings(tree):
    bindings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings.append((node.name, node.lineno))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bindings.append((node.id, node.lineno))
        elif isinstance(node, ast.arg):
            bindings.append((node.arg, node.lineno))
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            bindings.append((node.attr, node.lineno))
        elif isinstance(node, ast.alias) and node.asname:
            bindings.append((node.asname, node.lineno))
        elif isinstance(node, ast.keyword) and node.arg:
            bindings.append((node.arg, node.lineno))
    return bindings


def mask_literals(source):
    result = list(source)
    quote = ""
    index = 0
    while index < len(source):
        char = source[index]
        if quote:
            if char in {"\\", "`"} and index + 1 < len(source):
                result[index] = " "
                if source[index + 1] != "\n":
                    result[index + 1] = " "
                index += 2
                continue
            if source.startswith(quote, index):
                for offset in range(len(quote)):
                    result[index + offset] = " "
                index += len(quote)
                quote = ""
                continue
            if char != "\n":
                result[index] = " "
        elif char in {"'", '"'}:
            quote = char * 3 if source.startswith(char * 3, index) else char
            for offset in range(len(quote)):
                result[index + offset] = " "
            index += len(quote)
            continue
        index += 1
    return "".join(result)


def text_comments(source, suffix):
    masked = mask_literals(source)
    if suffix in {".html", ".htm"}:
        pattern = r"<!--"
    elif suffix == ".sql":
        pattern = r"--|/\*"
    elif suffix in {".js", ".ts", ".css", ".c", ".cpp", ".h", ".hpp"}:
        pattern = r"//|/\*"
    else:
        pattern = r"#"
    return [(masked.count("\n", 0, match.start()) + 1, "comment") for match in re.finditer(pattern, masked)]


def inspect_file(path):
    rel = path.relative_to(ROOT)
    issues = []
    if path.suffix.lower() in DOC_TYPES or path.suffix.lower() == ".txt":
        issues.append((0, "documentation"))
    if rel.parts[0] not in DATA_ROOTS:
        for part in rel.parts:
            name = Path(part).stem if "." in part else part
            if len(name) > NAME_LIMIT:
                issues.append((0, "name_limit:" + name))
    if path.suffix == ".py":
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(rel))
        if path.stem.startswith("test_"):
            issues.append((0, "sole_check"))
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                issues.append((token.start[0], "comment"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                issues.append((node.lineno, "documentation"))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                issues.append((node.lineno, "sole_check"))
            if isinstance(node, ast.Import) and any(name.name in {"pytest", "unittest"} for name in node.names):
                issues.append((node.lineno, "sole_check"))
            if isinstance(node, ast.ImportFrom) and node.module in {"pytest", "unittest"}:
                issues.append((node.lineno, "sole_check"))
        for name, line in name_bindings(tree):
            if len(name) > NAME_LIMIT:
                issues.append((line, "name_limit:" + name))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"execute", "executescript"} and node.args:
                sql = node.args[0]
                if isinstance(sql, ast.Constant) and isinstance(sql.value, str):
                    issues.extend((node.lineno + line - 1, kind) for line, kind in text_comments(sql.value, ".sql"))
    elif path.suffix.lower() in TEXT_CODE or path.name == ".gitignore":
        source = path.read_text(encoding="utf-8-sig")
        if rel.parts[0] in DATA_ROOTS and path.suffix.lower() in {".html", ".htm"}:
            return [{"path": rel.as_posix(), "line": line, "kind": kind} for line, kind in issues]
        issues.extend(text_comments(source, path.suffix.lower()))
        if path.suffix == ".ps1":
            for name in re.findall(r"\$([A-Za-z_][A-Za-z0-9_]*)", source):
                if len(name) > NAME_LIMIT and name != "ErrorActionPreference":
                    issues.append((0, "name_limit:" + name))
    return [{"path": rel.as_posix(), "line": line, "kind": kind} for line, kind in issues]


def main():
    issues = []
    files = owned_files()
    test_files = sorted(path.relative_to(ROOT).as_posix() for path in files if path.is_relative_to(ROOT / "tests") and path.suffix == ".py")
    if test_files != ["tests/are_comments.py"]:
        issues.append({"path": "tests", "line": 0, "kind": "sole_check"})
    for path in files:
        try:
            issues.extend(inspect_file(path))
        except (SyntaxError, UnicodeError, tokenize.TokenError) as error:
            issues.append({"path": path.relative_to(ROOT).as_posix(), "line": 0, "kind": str(error)})
    if issues:
        print(FAIL_TEXT, file=sys.stderr)
        print(json.dumps(issues, indent=2), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({"check": "Are there comments?", "comments": 0, "documents": 0, "tests": 1, "name_limit": NAME_LIMIT, "files": len(files)}))


if __name__ == "__main__":
    main()
