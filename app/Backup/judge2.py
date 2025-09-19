# =============================================================================
# File: app/judge.py
# Version: 2025-09-17 00:45:00 -03 (America/Sao_Paulo)
# Purpose:
#   - Cálculo de contribuição de cada fonte para uma resposta final (collab)
#   - Utilitários de similaridade textual (n-grams + âncoras)
#   - Juiz simples (heurístico) para comparações A vs B
#
# Uso principal:
#   from app.judge import contribution_ratio
#   ratios = contribution_ratio(final_answer, {"openai": openai_ans, "gemini": gemini_ans})
#
# Notas:
#   - Sem dependências externas; apenas stdlib.
#   - Evita regex recursivo (?R) — compatível com Python re.
# =============================================================================

from __future__ import annotations
import math
import re
import json
import unicodedata
from typing import Dict, List, Tuple, Iterable

# -----------------------------------------------------------------------------
# Normalização e tokenização
# -----------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

def _strip_accents(s: str) -> str:
    # mantém caracteres base (ú -> u)
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def _normalize(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = _strip_accents(text)
    text = text.lower()
    # mantém números e letras, remove pontuação; preserva espaços
    text = _PUNCT_RE.sub(" ", text)
    # normaliza múltiplos espaços
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _words(text: str) -> List[str]:
    t = _normalize(text)
    return t.split() if t else []

# -----------------------------------------------------------------------------
# N-grams e vetorização
# -----------------------------------------------------------------------------
def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 1:
        return [(t,) for t in tokens]
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(0, len(tokens) - n + 1)]

def _count_vector(items: Iterable[Tuple[str, ...]]) -> Dict[Tuple[str, ...], int]:
    d: Dict[Tuple[str, ...], int] = {}
    for it in items:
        d[it] = d.get(it, 0) + 1
    return d

def _cosine(a: Dict[Tuple[str, ...], int], b: Dict[Tuple[str, ...], int]) -> float:
    if not a or not b:
        return 0.0
    # produto escalar
    dot = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb:
            dot += va * vb
    # normas
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)

def _cosine_ngram_similarity(final_text: str, source_text: str) -> float:
    """
    Similaridade cosseno multi-n-gram:
      - tenta 5-grams; se pouco texto, cai para 4,3,2,1.
      - média ponderada dá mais peso a n maiores.
    """
    ftoks = _words(final_text)
    stoks = _words(source_text)

    # se muito curto, proteção
    if not ftoks or not stoks:
        return 0.0

    ns = [5, 4, 3, 2, 1]
    weights = {5: 0.35, 4: 0.25, 3: 0.2, 2: 0.12, 1: 0.08}

    # ajusta dinamicamente n máximo se os textos forem curtos
    max_n = min(5, len(ftoks), len(stoks))
    ns = [n for n in ns if n <= max_n]

    score = 0.0
    weight_sum = 0.0
    for n in ns:
        fvec = _count_vector(_ngrams(ftoks, n))
        svec = _count_vector(_ngrams(stoks, n))
        sim = _cosine(fvec, svec)
        w = weights.get(n, 0.1)
        score += w * sim
        weight_sum += w

    return score / weight_sum if weight_sum > 0 else 0.0

# -----------------------------------------------------------------------------
# Âncoras por frase (reforço)
# -----------------------------------------------------------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?\n])\s+")

def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # substitui quebras de linha múltiplas por simples antes do split
    text = re.sub(r"\n+", "\n", text).strip()
    # split por fim de sentença aproximado
    parts = _SENT_SPLIT_RE.split(text)
    # filtra sentenças não-vazias
    return [p.strip() for p in parts if p.strip()]

def _jaccard_words(a: str, b: str) -> float:
    wa = set(_words(a))
    wb = set(_words(b))
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    union = len(wa | wb)
    return inter / union if union else 0.0

def _anchor_boost(final_answer: str, source_text: str) -> float:
    """
    Reforça a similaridade com base em âncoras de frase:
      - Para cada sentença do final, busca a melhor sentença da fonte.
      - Se Jaccard(words) >= 0.6, conta como âncora forte.
      - Score = (#âncoras fortes / #sentenças finais) * 0.25  (peso máximo +0.25)
    Retorna valor em [0, 0.25].
    """
    final_sents = _split_sentences(final_answer)
    if not final_sents:
        return 0.0
    source_sents = _split_sentences(source_text)
    if not source_sents:
        return 0.0

    strong = 0
    for fs in final_sents:
        best = 0.0
        for ss in source_sents:
            j = _jaccard_words(fs, ss)
            if j > best:
                best = j
                if best >= 0.95:
                    break
        if best >= 0.6:
            strong += 1

    ratio = strong / max(1, len(final_sents))
    return min(0.25, 0.25 * ratio)

# -----------------------------------------------------------------------------
# Contribuição combinada
# -----------------------------------------------------------------------------
def _bounded(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def _combine_similarity(final_answer: str, source_text: str) -> float:
    """
    Combina similaridade de n-grams (base) com reforço por âncoras (boost).
    Retorna score em [0, 1].
    """
    base = _cosine_ngram_similarity(final_answer, source_text)  # [0,1]
    boost = _anchor_boost(final_answer, source_text)            # [0,0.25]
    return _bounded(base + boost, 0.0, 1.0)

def contribution_ratio(final_answer: str, sources: Dict[str, str]) -> Dict[str, float]:
    """
    Calcula a fração de contribuição de cada fonte para a resposta final.
    - final_answer: texto final fundido
    - sources: dict {nome_fonte: texto_fonte}

    Retorna: dict {nome_fonte: fração_em_[0,1]} somando 1.0.
    """
    if not final_answer or not sources:
        return {k: 0.0 for k in sources.keys()} if sources else {}

    # passo 1: score bruto por fonte
    raw_scores: Dict[str, float] = {}
    for name, src in sources.items():
        s = _combine_similarity(final_answer, src or "")
        raw_scores[name] = _bounded(s, 0.0, 1.0)

    total = sum(raw_scores.values())

    # passo 2: normaliza para somar 1.0 (se tudo 0 -> split uniforme)
    if total <= 1e-12:
        n = max(1, len(raw_scores))
        uniform = 1.0 / n
        return {k: uniform for k in raw_scores.keys()}

    return {k: v / total for k, v in raw_scores.items()}

# -----------------------------------------------------------------------------
# Juiz simples (opcional)
# -----------------------------------------------------------------------------
_BULLET_RE = re.compile(r"^\s*[-\*\u2022]", flags=re.MULTILINE)

def _structure_score(text: str) -> float:
    """
    Escore simples de estrutura/completude.
    Considera: tamanho moderado, presença de parágrafos e bullets.
    """
    if not text:
        return 0.0
    length = len(text)
    paras = len([p for p in text.split("\n\n") if p.strip()])
    bullets = len(_BULLET_RE.findall(text))
    # normalizações toscas para escala [0, 1]
    s_len = _bounded(length / 1000.0)          # cap em ~1000 chars
    s_par = _bounded(paras / 6.0)              # cap em ~6 parágrafos
    s_bul = _bounded(bullets / 8.0)            # cap em ~8 bullets
    # pesos
    return 0.5*s_len + 0.35*s_par + 0.15*s_bul

def judge_answers(a: str, b: str) -> Dict[str, str]:
    """
    Heurística simples: quem tiver melhor estrutura/completude vence.
    Retorna: {"winner": "A"|"B", "reason": "..."}
    """
    sa = _structure_score(a)
    sb = _structure_score(b)
    if abs(sa - sb) < 0.02:  # empate técnico
        win = "A" if len(a) >= len(b) else "B"
        return {"winner": win, "reason": "Empate técnico; maior completude por tamanho."}
    win = "A" if sa > sb else "B"
    return {"winner": win, "reason": f"Estrutura/completude: {sa:.2f} vs {sb:.2f}"}

# -----------------------------------------------------------------------------
# Utilitário seguro para extrair JSON (sem regex recursivo)
# -----------------------------------------------------------------------------
def safe_extract_json(text: str) -> Dict:
    """
    Tenta extrair o PRIMEIRO objeto JSON balanceando chaves manualmente.
    Útil quando modelos retornam texto + JSON.
    Retorna {} se não achar.
    """
    if not text:
        return {}
    s = text.strip()
    start = s.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = s[start:i+1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        break  # tenta próximo '{'
        start = s.find("{", start + 1)
    return {}
