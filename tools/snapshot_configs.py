#!/usr/bin/env python3
# =============================================================================
# File: tools/snapshot_configs.py
# Version: 2025-09-14 12:22:00 -03 (America/Sao_Paulo)
# Purpose:
#   Gera um snapshot consolidado dos arquivos de configuração do projeto
#   (com índice, metadados e conteúdo), para evitar perda de contexto e
#   detectar regressões/acidentes onde partes do arquivo somem.
#
# Uso:
#   python tools/snapshot_configs.py
#   python tools/snapshot_configs.py --output CONFIG_SNAPSHOT.md --manifest CONFIG_SNAPSHOT.manifest.json
#   python tools/snapshot_configs.py --include-ext .py .env .yaml .yml .toml .json .ini .cfg .conf --exclude-dirs app/modules
#
# Saídas padrão:
#   - CONFIG_SNAPSHOT.md               (consolidado com índice + conteúdo)
#   - CONFIG_SNAPSHOT.manifest.json    (manifesto com hash/nº linhas/mtime)
# =============================================================================

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

DEFAULT_OUTPUT = "CONFIG_SNAPSHOT.md"
DEFAULT_MANIFEST = "CONFIG_SNAPSHOT.manifest.json"

# extensões típicas de **configuração**
DEFAULT_INCLUDE_EXT = [
    ".py",       # settings.py, config.py etc (código-config)
    ".env", ".env.example",
    ".yaml", ".yml",
    ".toml",
    ".json",
    ".ini", ".cfg", ".conf",
    ".service",           # systemd
    ".properties",
    ".editorconfig",
]

# nomes de arquivos sem extensão que geralmente são config
DEFAULT_INCLUDE_NAMES = [
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Makefile",
    "Pipfile",
    "poetry.lock",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    ".gitignore",
    ".gitattributes",
    ".pre-commit-config.yaml",
    ".prettierrc",
    ".eslintrc", ".eslintrc.json", ".eslintrc.yml",
]

# diretórios a **excluir** (de módulos/artefatos/temporários)
DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".vscode",
    ".idea",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "site-packages",
    "venv",
    ".venv",
    "env",
    ".env",      # dir
    ".cache",
    ".ruff_cache",
    ".tox",
    ".coverage",
}

# paths completos a excluir (ajuste se precisar)
DEFAULT_EXCLUDE_PATHS = set()


def is_config_file(path: Path, include_ext: List[str], include_names: List[str]) -> bool:
    if not path.is_file():
        return False
    name = path.name
    suffix = path.suffix.lower()

    # match por nome inteiro
    if name in include_names:
        return True

    # match por extensão (case-insensitive)
    if suffix in [e.lower() for e in include_ext]:
        return True

    # arquivos .env.* (ex: .env.local)
    if name.startswith(".env."):
        return True

    return False


def iter_files(root: Path, exclude_dirs: Iterable[str]) -> Iterable[Path]:
    exclude_dirs_lower = {d.lower() for d in exclude_dirs}
    for dirpath, dirnames, filenames in os.walk(root):
        # prune diretórios
        dirnames[:] = [d for d in dirnames if d.lower() not in exclude_dirs_lower]
        # yield files
        for fn in filenames:
            yield Path(dirpath) / fn


def mtime_str(p: Path) -> str:
    ts = p.stat().st_mtime
    # timezone local do sistema (São Paulo no teu caso)
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_text_safely(p: Path) -> Tuple[str, int, str]:
    """
    Lê arquivo como texto (utf-8). Se falhar, tenta latin-1.
    Retorna: (conteudo, num_linhas, hash_sha256_hex)
    """
    raw: bytes
    try:
        raw = p.read_bytes()
    except Exception as e:
        # arquivos especiais podem falhar; retorna vazio para não quebrar
        return f"<<erro ao ler bytes: {e}>>", 0, sha256_of_bytes(b"")

    text: str
    for enc in ("utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            text = None  # type: ignore
    if text is None:
        # como último recurso, representação binária curta
        head = raw[:256]
        text = f"<<binário ({len(raw)} bytes). head: {head!r}>>"

    line_count = text.count("\n") + (0 if text.endswith("\n") else 1)
    return text, line_count, sha256_of_bytes(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera snapshot consolidado de arquivos de configuração.")
    parser.add_argument("--root", default=".", help="Diretório raiz do projeto (default: .)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Arquivo de saída (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help=f"Manifesto JSON (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--include-ext", nargs="*", default=DEFAULT_INCLUDE_EXT,
                        help="Extensões a incluir (ex.: .py .yml .toml ...)")
    parser.add_argument("--include-names", nargs="*", default=DEFAULT_INCLUDE_NAMES,
                        help="Nomes exatos de arquivos a incluir (ex.: Dockerfile Makefile ...)")
    parser.add_argument("--exclude-dirs", nargs="*", default=list(DEFAULT_EXCLUDE_DIRS),
                        help="Diretórios a excluir (nomes, não paths)")
    parser.add_argument("--exclude-paths", nargs="*", default=list(DEFAULT_EXCLUDE_PATHS),
                        help="Paths completos a excluir (começando na raiz). Ex.: app/modules configs/secrets")
    parser.add_argument("--max-bytes", type=int, default=0,
                        help="Se >0, trunca conteúdo por arquivo a este limite (segurança).")
    parser.add_argument("--sort", choices=["path", "mtime"], default="path",
                        help="Ordenação das seções: por path (default) ou mtime.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_md = Path(args.output).resolve()
    out_manifest = Path(args.manifest).resolve()

    # normaliza exclude paths
    exclude_paths_abs = { (root / p).resolve() for p in args.exclude_paths }

    candidates: List[Path] = []
    for p in iter_files(root, args.exclude_dirs):
        if any(str(p.resolve()).startswith(str(ex)) for ex in exclude_paths_abs):
            continue
        if is_config_file(p, args.include_ext, args.include_names):
            candidates.append(p)

    # ordenação
    if args.sort == "mtime":
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=False)
    else:
        candidates.sort(key=lambda p: str(p).lower())

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S %z") or dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entries = []
    total_lines = 0

    for p in candidates:
        content, line_count, sha = read_text_safely(p)
        if args.max_bytes and len(content.encode("utf-8", "ignore")) > args.max_bytes:
            # truncar mantendo info
            encoded = content.encode("utf-8", "ignore")
            content = encoded[: args.max_bytes].decode("utf-8", "ignore") + "\n<<TRUNCATED>>\n"
        entries.append({
            "path": str(p.relative_to(root)),
            "mtime": mtime_str(p),
            "lines": line_count,
            "sha256": sha,
            "content": content,
        })
        total_lines += line_count

    # manifesto JSON (para CI/automatização)
    manifest = {
        "generated_at": now,
        "root": str(root),
        "file_count": len(entries),
        "total_lines": total_lines,
        "hash_algorithm": "sha256",
        "files": [
            {k: v for k, v in e.items() if k != "content"} for e in entries
        ],
    }
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # MD consolidado com índice
    lines: List[str] = []
    lines.append("# CONFIG SNAPSHOT\n")
    lines.append(f"- Generated at: **{now}**\n")
    lines.append(f"- Root: `{root}`\n")
    lines.append(f"- Files: **{len(entries)}**\n")
    lines.append(f"- Total config lines: **{total_lines}**\n")
    lines.append("\n---\n")
    lines.append("## Index\n")
    if entries:
        width = len(str(len(entries)))
    else:
        width = 1
    for i, e in enumerate(entries, start=1):
        idx = str(i).rjust(width)
        lines.append(f"- [{idx}] `{e['path']}` — {e['lines']} lines — mtime {e['mtime']} — sha256 `{e['sha256'][:12]}…`")
    lines.append("\n---\n")

    for i, e in enumerate(entries, start=1):
        lines.append(f"## [{i}] {e['path']}")
        lines.append(f"- Last modified: **{e['mtime']}**")
        lines.append(f"- Lines: **{e['lines']}**")
        lines.append(f"- SHA-256: `{e['sha256']}`\n")
        # escolhe linguagem do bloco de código
        code_lang = ""
        suffix = Path(e["path"]).suffix.lower()
        if suffix in (".py",):
            code_lang = "python"
        elif suffix in (".yml", ".yaml"):
            code_lang = "yaml"
        elif suffix in (".json",):
            code_lang = "json"
        elif suffix in (".toml",):
            code_lang = "toml"
        elif suffix in (".ini", ".cfg", ".conf", ".properties"):
            code_lang = ""
        elif e["path"].endswith("Dockerfile"):
            code_lang = "dockerfile"
        elif e["path"].endswith("Makefile"):
            code_lang = "makefile"

        lines.append(f"```{code_lang}")
        lines.append(e["content"].rstrip("\n"))
        lines.append("```\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[ok] snapshot: {out_md}")
    print(f"[ok] manifest: {out_manifest}")
    print(f"[info] files: {len(entries)} | lines: {total_lines}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[aborted] interrupted by user", file=sys.stderr)
        raise
