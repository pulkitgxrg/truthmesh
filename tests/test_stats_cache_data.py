from pathlib import Path

import pytest

from cvsm import stats
from cvsm.config import SYSTEM_PRESETS, Switches
from cvsm.datasets import exact_match, f1_score, normalize_answer, split_ids
from cvsm.llm import LLMClient, Ledger, cost_usd
from cvsm.llm.base import FakeProvider, cache_key
from cvsm.types import ModelSpec


# ---------- statistics ----------
def test_mcnemar_exact():
    x = [1] * 10 + [0] * 2 + [1] * 5
    y = [0] * 10 + [1] * 2 + [1] * 5
    r = stats.mcnemar(x, y)
    assert (r["b"], r["c"]) == (10, 2)
    assert r["p_value"] == pytest.approx(2 * (1 + 12 + 66) / 4096)


def test_holm():
    r = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert r["a"]["reject"] and not r["c"]["reject"] and not r["b"]["reject"]
    assert r["a"]["p_adj"] == pytest.approx(0.03)
    assert r["c"]["p_adj"] == pytest.approx(0.06) and r["b"]["p_adj"] == pytest.approx(0.06)


def test_required_n():
    # 50 * ((1.95996 + 0.84162) * 0.04 / 0.05)^2 = 251.2 -> 252
    assert stats.required_n_from_se(0.04, 50, 0.05) == 252


def test_bootstrap_ci_contains_point():
    data = [(1, 4), (0, 5), (2, 3), (0, 2)] * 5
    p, lo, hi = stats.bootstrap_ci(data, stats.hr_of, n_boot=500)
    assert lo <= p <= hi and p == pytest.approx(3 / 14)


# ---------- cache ----------
def test_cache_roundtrip(tmp_path):
    fake = FakeProvider(lambda s, u: "hello " + u)
    m = ModelSpec("fake", "m1")
    llm = LLMClient(tmp_path, providers={"fake": fake})
    led = Ledger()
    r1 = llm.complete(m, "sys", "x", temperature=0.7, tag="run1", ledger=led, component="t")
    r2 = llm.complete(m, "sys", "x", temperature=0.7, tag="run1", ledger=led, component="t")
    r3 = llm.complete(m, "sys", "x", temperature=0.7, tag="run2", ledger=led, component="t")
    assert fake.n_calls == 2 and not r1.cached and r2.cached and not r3.cached
    assert r1.text == r2.text and r1.input_tokens == r2.input_tokens
    assert led.summary()["llm_calls"] == 3 and led.summary()["cached_calls"] == 1
    off = LLMClient(tmp_path, providers={"fake": fake}, offline=True)
    assert off.complete(m, "sys", "x", temperature=0.7, tag="run1").text == r1.text
    with pytest.raises(RuntimeError):
        off.complete(m, "sys", "new prompt", temperature=0.7, tag="run1")
    assert cache_key(m, "s", [{"role": "user", "content": "x"}], 0.0, 10, "") == \
        cache_key(m, "s", [{"role": "user", "content": "x"}], 0.0, 10, "")


def test_cost_known_and_unknown():
    assert cost_usd("anthropic:claude-haiku-4-5", 1_000_000, 1_000_000) == pytest.approx(6.0)
    assert cost_usd("fake:unknown", 10, 10) is None                     # never guessed


# ---------- datasets ----------
def test_em_f1_official_normalisation():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert exact_match("the Eiffel tower", "Eiffel Tower") == 1.0
    assert f1_score("Eiffel Tower in Paris", "Eiffel Tower") == pytest.approx(2 * 0.5 * 1 / 1.5)
    assert f1_score("yes", "no") == 0.0


def test_split_deterministic_and_disjoint():
    ids = [f"q{i}" for i in range(100)]
    s1, s2 = split_ids(ids, 13), split_ids(list(reversed(ids)), 13)
    assert s1 == s2 and len(s1["validation"]) == 20 and not set(s1["validation"]) & set(s1["test"])


@pytest.mark.skipif(not Path("data/raw/hotpot_qa_distractor_validation.parquet").exists(), reason="data not downloaded")
def test_hotpotqa_loader():
    from cvsm.datasets import load_hotpotqa
    exs = load_hotpotqa(Path("data"))
    assert len(exs) == 7405
    assert all(len(e.passages) >= 2 and e.gold_evidence for e in exs[:50])


# ---------- Table IV presets ----------
def test_presets_cover_table_iv():
    assert {f"b{i}" for i in range(1, 7)} | {f"a{i}" for i in range(1, 9)} | {"cvsm"} == set(SYSTEM_PRESETS)
    sw = Switches(**SYSTEM_PRESETS["a6"])
    assert not sw.correlation_discount
    sw = Switches(**SYSTEM_PRESETS["a8"])
    assert not sw.premise_cap and not sw.retraction


def test_budget_guard(tmp_path):
    from cvsm.budget import BudgetGuard
    g = BudgetGuard(1.0, tmp_path / "ledger.jsonl", fee_multiplier=1.1)
    assert g.allow()
    for _ in range(4):
        g.record(0.2, "query")
    assert g.spent() == pytest.approx(0.88)
    assert not g.allow()                    # 0.88 + 0.2*1.1 = 1.10 > 1.0
    assert BudgetGuard(None, tmp_path / "x.jsonl").allow()


def test_fever_metrics():
    from cvsm.task_metrics import fever_label_correct, fever_score, normalize_fever_label
    assert normalize_fever_label("Supported") == "SUPPORTS" and normalize_fever_label("not enough info") == "NOT ENOUGH INFO"
    assert fever_label_correct("refuted", "REFUTES") == 1
    gold_sets = [[("A", 1), ("B", 2)], [("C", 0)]]
    assert fever_score("SUPPORTS", [("C", 0)], "SUPPORTS", gold_sets) == 1          # second set covered
    assert fever_score("SUPPORTS", [("A", 1)], "SUPPORTS", gold_sets) == 0          # first set incomplete
    assert fever_score("REFUTES", [("C", 0)], "SUPPORTS", gold_sets) == 0           # wrong label
    assert fever_score("NOT ENOUGH INFO", [], "NOT ENOUGH INFO", []) == 1
