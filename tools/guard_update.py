#!/usr/bin/env python3
# =============================================================================
# File: tools/guard_update.py
# Version: 2025-09-14 13:45:00 -03 (America/Sao_Paulo)
# Purpose:
#   Garante que uma NOVA versão de arquivo (temp) não "encolha" nem perca
#   funções/classes públicas da versão ATUAL. Uso principal:
#   - Pre-commit/CI antes de aceitar substituição integral.
# Rules:
#   - NOVO.num_linhas >= ATUAL.num_linhas  (a menos que --allow-shrink)
#   - NOVO mantém TODAS funções/classes "públicas" (sem prefixo "_")
#   - NOVO preserva o bloco de header (primeiros comentários consecutivos)
# Exit codes:
#   0 = ok ; 1 = erro de regra ; 2 = erro de execução (I/O, parsing, etc.)
# =============================================================================

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Set, Tuple


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="latin-1")


def header_block(text: str) -> str:
    lines = text.splitlines()
    hdr = []
    for ln in lines:
        if ln.strip().startswith("#"):
            hdr.append(ln)
        else:
            break
    return "\n".join(hdr).strip()


def public_symbols_py(text: str) -> Tuple[Set[str], Set[str]]:
    """
    Retorna (funções_públicas, classes_públicas) a partir do AST.
    Definição de "público": nome não inicia com "_".
    """
    funcs: Set[str] = set()
    klass: Set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return funcs, klass

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            name = node.name
            if not name.startswith("_"):
                funcs.add(name)
        elif isinstance(node, ast.AsyncFunctionDef):
            name = node.name
            if not name.startswith("_"):
                funcs.add(name)
        elif isinstance(node, ast.ClassDef):
            name = node.name
            if not name.startswith("_"):
                klass.add(name)
    return funcs, klass


def main() -> int:
    ap = argparse.ArgumentParser(description="Guarda contra encolhimento/perda de símbolos ao substituir arquivo.")
    ap.add_argument("--current", required=True, help="Caminho do arquivo ATUAL (no repo).")
    ap.add_argument("--new", required=True, help="Caminho do arquivo NOVO (temp).")
    ap.add_argument("--allow-shrink", dest="allow_shrink", action="store_true",
                    help="Permite diminuir linhas (NÃO recomendado).")
    ap.add_argument("--lang", default="py", choices=["py", "any"], help="Heurística de símbolos (py=AST).")
    args = ap.parse_args()

    cur = Path(args.current)
    new = Path(args.new)

    if not cur.exists():
        print(f"[guard] current file not found: {cur}", file=sys.stderr)
        return 2
    if not new.exists():
        print(f"[guard] new file not found: {new}", file=sys.stderr)
        return 2

    cur_text = read_text(cur)
    new_text = read_text(new)

    cur_lines = cur_text.count("\n") + (0 if cur_text.endswith("\n") else 1)
    new_lines = new_text.count("\n") + (0 if new_text.endswith("\n") else 1)

    # Regra 1: não encolher (corrigido: usar args.allow_shrink)
    if not args.allow_shrink and new_lines < cur_lines:
        print(f"[guard][ERROR] line count decreased: {cur_lines} -> {new_lines} in {cur}", file=sys.stderr)
        return 1

    # Regra 2: manter header
    cur_hdr = header_block(cur_text)
    new_hdr = header_block(new_text)
    if cur_hdr and cur_hdr not in new_text:
        print(f"[guard][ERROR] header block missing or altered in {new}", file=sys.stderr)
        return 1

    # Regra 3: manter símbolos públicos (para .py)
    if args.lang == "py":
        cur_funcs, cur_classes = public_symbols_py(cur_text)
        new_funcs, new_classes = public_symbols_py(new_text)

        missing_funcs = sorted(cur_funcs - new_funcs)
        missing_classes = sorted(cur_classes - new_classes)

        if missing_funcs or missing_classes:
            if missing_funcs:
                print(f"[guard][ERROR] missing public functions in {new}: {', '.join(missing_funcs)}", file=sys.stderr)
            if missing_classes:
                print(f"[guard][ERROR] missing public classes in {new}: {', '.join(missing_classes)}", file=sys.stderr)
            return 1

    print(f"[guard][OK] {cur} -> {new} (lines {cur_lines}->{new_lines})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[aborted] interrupted by user", file=sys.stderr)
        raise

