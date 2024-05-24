import numpy as np
from fastapi import APIRouter, HTTPException, FastAPI
from fastapi.responses import JSONResponse
import os
import pandas as pd




# data = None





router = APIRouter()
app = FastAPI()

app.data = None


@router.post("/upload")
async def upload_file(filepath: str):
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    if not filepath.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File type not allowed")

    # filename = os.path.basename(filepath)
    # destination = os.path.join(UPLOAD_FOLDER, filename)

    #file to RAM
    app.data = pd.read_csv(filepath)
    app.data = app.data.iloc[1:]
    app.data.replace({np.nan: None}, inplace=True)

    # try:
    #     shutil.copy(filepath, destination)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Failed to copy file: {str(e)}")

    return {"message": "File successfully uploaded!"}



@router.get("/readfile")
async def read_file():
    if app.data.empty:
        raise HTTPException(status_code=500, detail="No file found!")
    
    try:
        # zwracamy tylko 20 pierwszych elementów
        result = app.data.head(20).to_json()
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
    if normalization == "other":
        pass
    else:
        raise HTTPException(status_code=400, detail=f"Normalization {normalization} not implemented!!")

    app.data = normalized_data

    return read_file()


@router.put("/pca")
async def pca_analysis():
    if app.data.empty:
        raise HTTPException(status_code=500, detail="No file found!")
    
    return read_file()



app.include_router(router)








