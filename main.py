from fastapi import FastAPI
from model import predict_box
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

class image_base64(BaseModel):
    img_base64: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
def read_image(file: image_base64):
    base64 = file.img_base64
    a = predict_box(base64)
    return {"result": a}


