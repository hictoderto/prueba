# backend/main.py
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/saludo")
def saludo(nombre: str):
    return {"mensaje": f"Hola {nombre}, soy Python!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
