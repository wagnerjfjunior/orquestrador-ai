#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-https://orquestrador-ai.onrender.com}"
REQ_ID="abc123xyz"

echo "=== E2E check contra: ${BASE_URL}"
echo

echo "1) GET /health"
curl -sS -i "${BASE_URL}/health" | sed -n '1,6p'
echo

echo "2) GET /ready"
curl -sS -i "${BASE_URL}/ready" | sed -n '1,6p'
echo

echo "3) X-Request-ID propagado"
curl -sS -i -H "x-request-id: ${REQ_ID}" "${BASE_URL}/ready" | sed -n '1,10p' | grep -i "x-request-id" || {
  echo "ERRO: header X-Request-ID não encontrado na resposta"
  exit 1
}
echo

echo "4) /metrics (head -n 20)"
curl -sS "${BASE_URL}/metrics" | head -n 20
echo

echo "5) /ask echo"
RESP=$(curl -sS -X POST "${BASE_URL}/ask?provider=echo" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello render"}')
echo "${RESP}" | jq . 2>/dev/null || echo "${RESP}"
echo

echo "6) Métrica de sucesso do echo (ask_requests_total)"
curl -sS "${BASE_URL}/metrics" | grep 'ask_requests_total{provider="echo",status="success"}' || true
echo

echo "7) Forçando erro de provider (openai sem chave)"
STATUS=$(curl -sS -o /tmp/openai_err.json -w "%{http_code}" -X POST "${BASE_URL}/ask?provider=openai" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"qualquer"}' || true)
echo "HTTP ${STATUS}"
cat /tmp/openai_err.json; echo

echo "8) Métrica de erro do openai"
curl -sS "${BASE_URL}/metrics" | grep 'ask_requests_total{provider="openai",status="error"}' || true
echo

echo "=== FIM E2E ✅"
