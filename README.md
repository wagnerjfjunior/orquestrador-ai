# Orquestrador AI (Esqueleto)
- FastAPI
- Endpoints:
  - GET /health -> {"status":"ok"}
  - POST /ask   -> {"message":"Hello World", "echo":{...}}

## Rodar local
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload --port 8080
