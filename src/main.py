# To Load WebApp:
# uvicorn --host 0.0.0.0 main:app --reload
# uvicorn main:app --reload --port 8000
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from GRU_Build import plotting

app = FastAPI()

origins = ["*",
           "http://localhost:3000/",
           "http://localhost/",
           "https://localhost:3000/",
           "https://localhost/"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.get("/predict")
def return_prediction(input_str: str):
    image = plotting()
    print("hi")
    prediction = "hi"
    return {"prediction": prediction}
