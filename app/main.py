from fastapi import FastAPI, Body
app = FastAPI(title="Orquestrador AI", version="0.1.0")
@app.get("/health")
async def health():
return {"status": "ok"}

@app.post("/ask")
async def ask(payload: dict = Body(default={})):
return {"message": "Hello World", "echo": payload}
