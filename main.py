from fastapi import FastAPI
from csv_service import router

app = FastAPI()
app.include_router(router)