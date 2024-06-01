import numpy as np
from fastapi import APIRouter, HTTPException, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.io as pio
from starlette.responses import StreamingResponse
import datetime 
from sklearn.cluster import DBSCAN, AgglomerativeClustering

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


#GLOBAL VARIABLES
app.data = None
app.pca_data = None
app.cluster_data = None





def detect_and_parse(contents: str):
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')), delimiter=';')

    if len(df.columns) == 1 and ',' in df.iloc[0, 0]:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')), delimiter=',')

    return df

def string_to_date_to_number(date_string):
    return datetime.datetime.strptime(date_string, "%d.%m.%Y %H:%M").timestamp()





@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()

    df = detect_and_parse(contents)

    df = df.dropna()
    df = df[~df.isin([float('inf'), float('-inf')]).any(axis=1)]

    app.data = df

    data = df.to_dict(orient="records")
    columns = list(df.columns)

    return {"filename": file.filename, "data": data, "columns": columns }


@router.get("/readfile")
async def read_file(n_of_rows: int):
    if app.data.empty:
        raise HTTPException(status_code=500, detail="No data found!")
    
    try:
        if n_of_rows > 0:
            return JSONResponse(content=app.data.head(n_of_rows).to_dict(orient='records'))
        return JSONResponse(content=app.to_dict(orient='records'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download")
async def download_file():
    if app.data.empty:
        raise HTTPException(status_code=500, detail="No data found!")

    try:
        csv_content = app.data.to_csv(index=False).encode('utf-8')
        return StreamingResponse(iter([csv_content]), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=data.csv"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/rename")
async def rename_variables(old_name: str, new_name: str):
    if app.data.empty:
        raise HTTPException(status_code=500, detail="No data found!")
    
    try:
        # Zmiana nazwy kolumny
        app.data.rename(columns={old_name: new_name}, inplace=True)

        return {"message": f"Variable '{old_name}' renamed to '{new_name}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.put("/convert")
async def convert_to_numeric():
    #conversion to numeric values
    for column in app.data.columns:
        if app.data[column].dtype == 'object':
            try:
                app.data[column] = pd.to_numeric(app.data[column].str.replace(',', '.'))
            except:
                app.data[column] = app.data[column].map(string_to_date_to_number)
    return {"message": f"Conversion completed successfully!"}


@router.put("/normalize")
async def normalize_data(normalization: str):
    if app.data.empty:
        raise HTTPException(status_code=500, detail="No data found!")
    
    normalized_data = app.data.copy()

    if normalization == "minmax":
        for column in normalized_data.columns:
            if normalized_data[column].dtype != np.number:
                continue
            normalized_data[column] = (normalized_data[column] - normalized_data[column].min()) / (normalized_data[column].max() - normalized_data[column].min())      
    elif normalization == "standarization":
        for column in normalized_data.columns:
            if normalized_data[column].dtype != np.number:
                continue
            normalized_data[column] = (normalized_data[column] - normalized_data[column].mean()) / normalized_data[column].std()
    elif normalization == "log":
        for column in normalized_data.columns:
            if normalized_data[column].dtype != np.number or normalized_data[column].min() <= 0:
                continue
            normalized_data[column] = np.log(normalized_data[column])
    else:
        raise HTTPException(status_code=400, detail=f"Normalization {normalization} not implemented!!")

    app.data = normalized_data

    return {"message": f"Normalization '{normalization}' completed successfully!"}


@router.put("/pca")
async def pca_analysis(n_components: int):
    if app.data is None or app.data.empty:
        raise HTTPException(status_code=500, detail="No data found!")

    numeric_data = app.data.select_dtypes(include=[np.number])

    if numeric_data.shape[1] < n_components:
        raise HTTPException(status_code=400,
                            detail="Number of components is greater than the number of numeric features")

    try:
        pca = PCA()
        components = pca.fit_transform(numeric_data)
        labels = {str(i): f"PC {i+1}" for i in range(n_components)}

        result = {
            "components" : components.tolist(),
            "labels" : labels,
            "dimensions" : [i for i in range(n_components)],
            "explained_variance" : pca.explained_variance_ratio_.sum()
        }

        app.pca_data = result

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@router.get("/pca/visualization")
async def pca_visulalization():
    if app.pca_data is None:
        raise HTTPException(status_code=500, detail="No PCA data found!")
    
    fig = px.scatter_matrix(
        app.pca_data["components"],
        labels=app.pca_data["labels"],
        dimensions=app.pca_data["dimensions"],
    )

    fig.update_traces(diagonal_visible=False)
    png_bytes = pio.to_image(fig, format="png")

    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@router.put("/kMeans")
async def clustering_kMeans(n_clusters: int):
    if app.data is None or app.data.empty:
        raise HTTPException(status_code=500, detail="No data found!")

    kmeans = KMeans(n_clusters=n_clusters)
    app.cluster_data = kmeans.fit_predict(app.data)
    
    return {"message": f"Clustering 'kMeans' completed successfully!"}

@router.put("/DBSCAN")
async def clustering_DBSCAN(eps: float, min_samples: int):
    if app.data is None or app.data.empty:
        raise HTTPException(status_code=500, detail="No data found!")

    # NOT TESTED!!!
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    app.cluster_data = dbscan.fit_predict(app.data)

    return {"message": f"Clustering 'DBSCAN' completed successfully!"}


@router.put("/agglomerative")
async def clustering_agglomerative(n_clusters: int):
    if app.data is None or app.data.empty:
        raise HTTPException(status_code=500, detail="No data found!")

    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    app.cluster_data = agglomerative.fit_predict(app.data)


    return {"message": f"Agglomerative Clustering completed successfully!"}


@router.get("/cluster_visualization")
async def cluster_visualization():
    if app.cluster_data is None:
        raise HTTPException(status_code=500, detail="No clustering data found!")

    # TODO


@router.get("/cluster_stats/{cluster_id}")
async def cluster_statistics(cluster_id: int):
    if app.cluster_data is None:
        raise HTTPException(status_code=500, detail="No clustering data found!")

    try:
        cluster_series = pd.Series(app.cluster_data, index=app.data.index)  # Ensure the same index
        cluster_data = app.data[cluster_series == cluster_id].reset_index(drop=True)  # Reset index
        stats = cluster_data.describe()
        return JSONResponse(content=stats.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#include router
app.include_router(router)








