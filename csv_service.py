import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
import pandas as pd
import shutil

router = APIRouter()

UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@router.post("/upload")
async def upload_file(filepath: str):
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    if not filepath.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File type not allowed")

    filename = os.path.basename(filepath)
    destination = os.path.join(UPLOAD_FOLDER, filename)

    try:
        shutil.copy(filepath, destination)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to copy file: {str(e)}")

    return {"message": "File successfully copied", "filename": filename}

@router.get("/readfile")
async def read_file(filename: str):
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        df = pd.read_csv(filepath)

        # Zastąpienie wartości "nan" wartością None
        df.replace({np.nan: None}, inplace=True)

        # zwracamy tylko 20 pierwszych elementów
        result = df.head(20).to_dict(orient='records')
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/rename")
async def rename_variables(filename: str, old_name: str, new_name: str):
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        df = pd.read_csv(filepath)
        if old_name not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{old_name}' does not exist")

        # Zmiana nazwy kolumny
        df.rename(columns={old_name: new_name}, inplace=True)

        # Zapisanie zmienionego pliku
        df.to_csv(filepath, index=False)

        return {"message": f"Variable '{old_name}' renamed to '{new_name}' in file '{filename}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

