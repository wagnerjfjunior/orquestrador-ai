# app/judge.py
from __future__ import annotations
from typing import Dict, Any
from app.observability import logger
from app.openai_client import is_configured as openai_configured, ask_openai
from app.gemini_client import is_configured as gemini_configured, ask_gemini
import json

JUDGE_PROMPT = """Você é um avaliador técnico. Receba uma PERGUNTA e duas RESPOSTAS (A e B).
Analise CORREÇÃO, CLAREZA, SEGURANÇA e UTILIDADE.

Saída (JSON estrito, UMA linha):
{"winner":"A|B|tie","reason":"<explicação curta>"}

PERGUNTA:
{question}

RESPOSTA A:
{answer_a}

RESPOSTA B:
{answer_b}
"""

def _parse_simple_json_line(txt: str) -> Dict[str, Any]:
    line = (txt or "").strip()
    start = line.find("{")
    end = line.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(line[start:end+1])
            w = (obj.get("winner") or "tie").lower()
            if w not in ("a", "b", "tie"):
                w = "tie"
            return {"winner": w, "reason": obj.get("reason") or ""}
        except Exception:
            pass
    return {"winner": "tie", "reason": (line[:240] if line else "")}

def judge_answers(question: str, answer_a: str, answer_b: str) -> Dict[str, Any]:
    prompt = JUDGE_PROMPT.format(question=question, answer_a=answer_a, answer_b=answer_b)

    if openai_configured():
        try:
            logger.info("judge.start", provider="openai")
            resp = ask_openai(prompt)
            parsed = _parse_simple_json_line(resp.get("answer"))
            logger.info("judge.done", provider="openai", winner=parsed["winner"])
            return {"provider": "openai", **parsed}
        except Exception as e:
            logger.info("judge.openai_error", error=str(e))

    if gemini_configured():
        try:
            logger.info("judge.start", provider="gemini")
            resp = ask_gemini(prompt)
            parsed = _parse_simple_json_line(resp.get("answer"))
            logger.info("judge.done", provider="gemini", winner=parsed["winner"])
            return {"provider": "gemini", **parsed}
        except Exception as e:
            logger.info("judge.gemini_error", error=str(e))

    # Heurística
    a_len = len((answer_a or "").strip())
    b_len = len((answer_b or "").strip())
    if a_len == 0 and b_len == 0:
        return {"provider": "heuristic", "winner": "tie", "reason": "Ambas vazias."}
    if abs(a_len - b_len) < 10:
        return {"provider": "heuristic", "winner": "tie", "reason": "Tamanho similar; empate técnico."}
    winner = "a" if a_len > b_len else "b"
    return {"provider": "heuristic", "winner": winner, "reason": "Resposta mais completa (comprimento)."}
