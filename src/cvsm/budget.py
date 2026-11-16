"""Study-wide spend tracking (list-price USD) with a hard stop before the budget would be exceeded.

Every completed query (system + instrumentation) and every evaluation appends a row to the spend ledger.
`BudgetGuard.allow()` refuses to start a query if spent + (running mean cost of a query) > budget."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class BudgetGuard:
    def __init__(self, budget_usd: float | None, ledger_path: str | Path = "runs/spend_ledger.jsonl",
                 fee_multiplier: float = 1.0):
        self.budget = budget_usd
        self.path = Path(ledger_path)
        self.fee = fee_multiplier          # e.g. 1.10 for a gateway that adds 10% on top of list price
        self._lock = threading.Lock()
        self._recent: list[float] = []

    def spent(self) -> float:
        if not self.path.exists():
            return 0.0
        return sum(json.loads(l)["usd"] for l in self.path.read_text().splitlines() if l.strip()) * self.fee

    def record(self, usd: float | None, what: str, run: str = "") -> None:
        if usd is None:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "usd": usd, "what": what,
                                    "run": run}) + "\n")
            if what == "query":
                self._recent = (self._recent + [usd])[-50:]

    def allow(self) -> bool:
        if self.budget is None:
            return True
        with self._lock:
            nxt = (sum(self._recent) / len(self._recent)) if self._recent else 0.0
            return self.spent() + nxt * self.fee <= self.budget
