import uvicorn
from fastapi import FastAPI, File
from model import predict_box
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


@app.post("/")
def read_image(file: bytes = File(...)):
    a = predict_box(file)
    return {"result": a}


