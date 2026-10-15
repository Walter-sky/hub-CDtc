"""Check that the course libraries can be imported.

Any installed version is fine. The script passes when the base
Miniconda Python environment can import every library listed in:
https://note.youdao.com/s/BAcOYrOB
"""

import importlib
import sys

# import name shown in the course notes
REQUIRED = [
    "numpy",
    "pandas",
    "matplotlib",
    "torch",
    "gensim",
    "sklearn",
    "peft",
    "transformers",
]


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    print(f"Executable {sys.executable}")
    print("Import check for https://note.youdao.com/s/BAcOYrOB")
    print("-" * 60)

    missing = []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            missing.append(name)
            print(f"MISSING  {name:<16} ({exc.__class__.__name__}: {exc})")
            continue

        version = getattr(module, "__version__", "imported")
        print(f"OK       {name:<16} {version}")

    print("-" * 60)
    if missing:
        print("These libraries could not be imported: " + ", ".join(missing))
        return 1

    print("All required libraries can be imported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
