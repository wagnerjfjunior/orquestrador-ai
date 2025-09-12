from fastapi import FastAPI, Body

# Inicializa o app FastAPI
app = FastAPI(title="Orquestrador AI", version="0.1.0")

# Rota raiz (para evitar "Not Found" ao acessar diretamente o domínio)
@app.get("/")
async def root():
    return {"service": "orquestrador-ai", "status": "live"}

# Healthcheck (para monitoramento e testes)
@app.get("/health")
async def health():
    return {"status": "ok"}

# Endpoint de teste para validação da pipeline
@app.post("/ask")
async def ask(payload: dict = Body(default={})):
    return {"message": "Hello World", "echo": payload}
