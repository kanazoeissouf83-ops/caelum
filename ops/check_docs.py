"""Check the documentation: valid docs.json, and every ```python block compiles.

Run from the repository root:  python ops/check_docs.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def main() -> int:
    errors: list[str] = []
    config = json.loads((DOCS / "docs.json").read_text(encoding="utf-8"))
    for group in config["navigation"]["groups"]:
        for page in group["pages"]:
            if not (DOCS / f"{page}.mdx").exists():
                errors.append(f"docs.json references a missing page: {page}")
    blocks = 0
    for page in sorted(DOCS.glob("*.mdx")):
        for index, match in enumerate(BLOCK.finditer(page.read_text(encoding="utf-8"))):
            blocks += 1
            source = match.group(1).replace("[...]", "[]")
            try:
                compile(source, f"{page.name}#{index}", "exec")
            except SyntaxError as error:
                errors.append(f"{page.name} block {index}: {error}")
    for error in errors:
        print("ERROR", error)
    print(f"{blocks} python blocks checked, {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
