# =============================================================================
# File: app/tests/test_judge.py
# Purpose: Testes mínimos de mesa para app/judge.py
# Run:
#   pytest -q
# =============================================================================

import math
from app.judge import contribution_ratio, judge_answers

FINAL = """Entropia é uma medida de desordem e de dispersão de energia.
Em sistemas isolados, a entropia tende a aumentar (Segunda Lei).
"""

OPENAI = """A entropia mede a desordem e a distribuição de energia de um sistema.
A segunda lei da termodinâmica afirma que a entropia de um sistema isolado tende
a aumentar com o tempo.
"""

GEMINI = """Pense em uma sala arrumada (baixa entropia) e depois bagunçada (alta entropia).
Em sistemas isolados, a entropia tende a crescer ao longo do tempo segundo a Segunda Lei.
"""

def test_contribution_sums_to_one():
    ratios = contribution_ratio(FINAL, {"openai": OPENAI, "gemini": GEMINI})
    total = sum(ratios.values())
    assert math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6)

def test_contributions_positive_when_overlap():
    ratios = contribution_ratio(FINAL, {"openai": OPENAI, "gemini": GEMINI})
    assert ratios["openai"] > 0.0
    assert ratios["gemini"] > 0.0

def test_contribution_uniform_when_no_overlap():
    final = "Texto sem relação alguma."
    srcs = {"openai": "aaaa bbbb", "gemini": "cccc dddd"}
    ratios = contribution_ratio(final, srcs)
    # sem sobreposição, cai no split uniforme
    assert math.isclose(ratios["openai"], 0.5, rel_tol=1e-6)
    assert math.isclose(ratios["gemini"], 0.5, rel_tol=1e-6)

def test_judge_returns_valid_winner_key():
    verdict = judge_answers(OPENAI, GEMINI)
    assert verdict["winner"] in ("A", "B")
    assert isinstance(verdict["reason"], str) and len(verdict["reason"]) > 0
