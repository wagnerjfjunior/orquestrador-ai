







# app/judge.py
from __future__ import annotations
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------
# Normalização e tokenização
# ---------------------------------------------------------------------
def _normalize(text: str) -> str:
    if not text:
        return ""
    # minúsculas
    t = text.lower()
    # espaços uniformes
    t = " ".join(t.split())
    return t

def _tokens(text: str) -> List[str]:
    t = _normalize(text)
    return t.split() if t else []

# ---------------------------------------------------------------------
# N-grams e similaridade de Jaccard
# ---------------------------------------------------------------------
def _ngram_set(tokens: List[str], n: int = 3) -> set:
    if n <= 1:
        return set(tokens)
    if not tokens or len(tokens) < n:
        return set()
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def jaccard(a: str, b: str, n: int = 3) -> float:
    ta, tb = _tokens(a), _tokens(b)
    A, B = _ngram_set(ta, n), _ngram_set(tb, n)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

# ---------------------------------------------------------------------
# Heurística de “voto” (usada em heuristic/crossvote)
# ---------------------------------------------------------------------
def choose_winner_len(a: str, b: str) -> str:
    """
    Critério simples e determinístico: vence a resposta mais longa.
    Empate favorece 'openai' para estabilidade.
    """
    la, lb = len(a or ""), len(b or "")
    if la >= lb:
        return "openai"
    return "gemini"

# ---------------------------------------------------------------------
# Fusão colaborativa (collab)
# ---------------------------------------------------------------------
def _split_paragraphs(s: str) -> List[str]:
    if not s:
        return []
    parts = [p.strip() for p in s.strip().split("\n\n")]
    return [p for p in parts if p]

def collab_fuse(source_answers: Dict[str, str]) -> str:
    """
    Junta parágrafos das fontes (openai/gemini), removendo duplicatas
    via Jaccard n-grams e adicionando um rodapé com as fontes usadas.
    """
    # Colete todos os parágrafos, anotando a origem
    paras: List[Tuple[str, str]] = []
    for prov, text in (source_answers or {}).items():
        for p in _split_paragraphs(text or ""):
            if p:
                paras.append((prov, p))

    # Remoção de duplicatas aproximadas (limiar 0.75)
    fused: List[str] = []
    kept_idx: List[int] = []
    for i, (_, p_i) in enumerate(paras):
        keep = True
        for j in kept_idx:
            _, p_j = paras[j]
            if jaccard(p_i, p_j, n=3) >= 0.75:
                keep = False
                break
        if keep:
            kept_idx.append(i)
            fused.append(p_i)

    # Rodapé com fontes utilizadas
    used = [prov for prov, txt in (source_answers or {}).items() if (txt or "").strip()]
    if used:
        fused.append(" ".join(f"[Fonte: {u}]" for u in used))

    return "\n\n".join(fused).strip()

# ---------------------------------------------------------------------
# Estimativa de contribuição por fonte
# ---------------------------------------------------------------------
def contribution_ratio(final_answer: str, sources: Dict[str, str]) -> Dict[str, float]:
    """
    Para cada parágrafo do final, mede a maior similaridade (Jaccard de n-grams)
    com algum parágrafo de cada fonte. A média por fonte é normalizada para somar 1.0.
    """
    final_paras = _split_paragraphs(final_answer)
    if not final_paras or not sources:
        return {k: 0.0 for k in (sources or {}).keys()}

    sims: Dict[str, float] = {k: 0.0 for k in sources.keys()}

    for k, v in sources.items():
        src_paras = _split_paragraphs(v or "")
        if not src_paras:
            sims[k] = 0.0
            continue
        total = 0.0
        for fp in final_paras:
            best = 0.0
            for sp in src_paras:
                best = max(best, jaccard(fp, sp, n=3))
            total += best
        sims[k] = total / max(1, len(final_paras))

    s = sum(sims.values())
    if s <= 0:
        return {k: 0.0 for k in sims.keys()}
    return {k: round(v / s, 4) for k, v in sims.items()}

__all__ = [
    "jaccard",
    "choose_winner_len",
    "collab_fuse",
    "contribution_ratio",
]
