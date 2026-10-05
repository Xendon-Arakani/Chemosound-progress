import py_compile
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    guard = runpy.run_path(str(ROOT / "tests" / "are_comments.py"))
    guard["main"]()
    for path in guard["owned_files"]():
        if path.suffix == ".py":
            py_compile.compile(str(path), doraise=True)


if __name__ == "__main__":
    main()
