import numpy as np
from fastapi import APIRouter, HTTPException, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import io
import csv
import pandas as pd


router = APIRouter()
app = FastAPI()

origins = [
    "http://localhost:5173",
    "localhost:5173"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
default_number_of_rows = 10

#global variable for given data
app.data = None

def detect_and_parse(contents: str):
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')), delimiter=';')

    if len(df.columns) == 1 and ',' in df.iloc[0, 0]:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')), delimiter=',')

    return df

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()

    df = detect_and_parse(contents)

    df = df.dropna()
    df = df[~df.isin([float('inf'), float('-inf')]).any(axis=1)]

    data = df.to_dict(orient="records")
    columns = list(df.columns)

    app.data = data

    return {"filename": file.filename, "data": data, "columns": columns }


@router.get("/readfile")
async def read_file(n_of_rows: int):
    if app.data.empty:
        raise HTTPException(status_code=500, detail="No file found!")
    
    try:
        result = app.data.head(n_of_rows).to_json(orient='records', lines=True)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.put("/rename")
async def rename_variables(old_name: str, new_name: str):
    if app.data.empty:
        raise HTTPException(status_code=500, detail="No file found!")
    
    try:
        # Zmiana nazwy kolumny
        app.data.rename(columns={old_name: new_name}, inplace=True)

        return {"message": f"Variable '{old_name}' renamed to '{new_name}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@router.put("/normalize")
async def normalize_data(normalization: str):
    if app.data.empty:
        raise HTTPException(status_code=500, detail="No file found!")
    
    normalized_data = app.data.copy()

    if normalization == "minmax":
        for column in normalized_data.columns:
            normalized_data[column] = (normalized_data[column] - normalized_data[column].min()) / (normalized_data[column].max() - normalized_data[column].min())      
    elif normalization == "other":
        pass
    else:
        raise HTTPException(status_code=400, detail=f"Normalization {normalization} not implemented!!")

    app.data = normalized_data

    # return read_file(default_number_of_rows)
    return {"message": f"Normalization completed successfully!"}


@router.put("/pca")
async def pca_analysis():
    if app.data.empty:
        raise HTTPException(status_code=500, detail="No file found!")
    
    return {"message": f"PCA completed successfully!"}




#include router
app.include_router(router)








