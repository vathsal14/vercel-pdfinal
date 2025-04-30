from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.predict import app as predict_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
