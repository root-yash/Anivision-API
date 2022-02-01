from fastapi import FastAPI
from model import predict_box
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from urllib.request import urlretrieve
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
async def read_image(file: image_base64):
    #download model file
    if os.path.exists('Model/model_398_trace.pt') == False:
        url = 'https://onedrive.live.com/download?cid=470A5A8DB59AAEA1&resid=470A5A8DB59AAEA1%2114267&authkey=AMbqal_UGof27TE'
        filename = 'Model/model_398_trace.pt'
        urlretrieve(url, filename)
    base64 = file.img_base64
    a = predict_box(base64)
    return {"result": a}


