from __future__ import annotations

import json
import re


def parse_json(text: str):
    """Parse the first JSON object/array in `text` (tolerates code fences and surrounding prose)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = t.find(open_c)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(t)):
                ch = t[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == open_c:
                    depth += 1
                elif ch == close_c:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(t[start:i + 1])
                        except json.JSONDecodeError:
                            break
            start = t.find(open_c, start + 1)
    raise ValueError(f"no JSON found in: {text[:200]!r}")


_ANS = re.compile(r"ANSWER\s*:\s*(.*?)(?:\n|$)", re.I)
_EXP = re.compile(r"EXPLANATION\s*:\s*(.*)", re.I | re.S)


def parse_answer(text: str) -> tuple[str, str]:
    a = _ANS.search(text)
    e = _EXP.search(text)
    ans = a.group(1).strip() if a else ""
    exp = e.group(1).strip() if e else ("" if a else text.strip())
    return ans, exp
