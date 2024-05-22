import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
import pandas as pd
import shutil

router = APIRouter()

UPLOAD_FOLDER = 'uploads'

data = None


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

    #file to RAM
    data = pd.read_csv(filepath)
    data.replace({np.nan: None}, inplace=True)

    try:
        shutil.copy(filepath, destination)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to copy file: {str(e)}")

    return {"message": "File successfully copied", "filename": filename}

@router.get("/readfile")
async def read_file():
    if data == None:
        raise HTTPException(status_code=500, detail="No file found!")
    
    try:
        # zwracamy tylko 20 pierwszych elementów
        result = data.head(20).to_dict(orient='records')
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/rename")
async def rename_variables(old_name: str, new_name: str):
    if data == None:
        raise HTTPException(status_code=500, detail="No file found!")
    
    try:
        # Zmiana nazwy kolumny
        data.rename(columns={old_name: new_name}, inplace=True)

        return {"message": f"Variable '{old_name}' renamed to '{new_name}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.put("/normalize")
async def normalize_data(normalization: str):
    raise HTTPException(status_code=500, detail=str(e))

