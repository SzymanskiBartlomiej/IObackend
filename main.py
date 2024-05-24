import numpy as np
from fastapi import APIRouter, HTTPException, FastAPI
from fastapi.responses import JSONResponse
import os
import io
import pandas as pd


router = APIRouter()
app = FastAPI()


default_number_of_rows = 10

#global variable for given data
app.data = None


@router.post("/upload")
async def upload_file(filepath: str):
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    if not filepath.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    data_string = ""
    with open(filepath, 'r') as file:
        data_string = file.read().replace(',', '.')

    #file to RAM
    app.data = pd.read_csv(io.StringIO(data_string), sep=";")

    #replace , -> . in strings
    # app.data.map(lambda string: string.replace(",", "."))

    app.data.replace({np.nan: None}, inplace=True)

    #conversion to numeric values
    for column in app.data.columns:
        try:
            app.data[column] = pd.to_numeric(app.data[column])
        except:
            app.data[column] = pd.to_datetime(app.data[column])

    return {"message": "File successfully uploaded!"}



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








