import numpy as np
from fastapi import APIRouter, HTTPException, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift, AffinityPropagation
import plotly.express as px
import plotly.io as pio
from starlette.responses import StreamingResponse
import datetime
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.stats import kurtosis, skew

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

# GLOBAL VARIABLES
app.data = None
app.data_before_selection = None
app.pca_data = None
app.cluster_data = None
app.kMeans_data = None
app.DBSCAN_data = None
app.numeric_data = None
app.normalized_data = None


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

    return {"filename": file.filename, "data": data, "columns": columns}


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
async def download_file(step: int):
    if step == 1 and app.data is not None and not app.data.empty:
        try:
            csv_content = app.data.to_csv(index=False).encode('utf-8')
            return StreamingResponse(iter([csv_content]), media_type="text/csv",
                                     headers={"Content-Disposition": "attachment; filename=data.csv"})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    if step == 2 and app.normalized_data is not None and not app.normalized_data.empty:
        try:
            csv_content = app.normalized_data.to_csv(index=False).encode('utf-8')
            return StreamingResponse(iter([csv_content]), media_type="text/csv",
                                     headers={"Content-Disposition": "attachment; filename=data.csv"})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    if step == 3 and app.cluster_data is not None and not app.cluster_data.empty:
        try:
            csv_content = app.cluster_data.to_csv(index=False).encode('utf-8')
            return StreamingResponse(iter([csv_content]), media_type="text/csv",
                                     headers={"Content-Disposition": "attachment; filename=data.csv"})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=500, detail="No data found!")


@router.put("/rename")
async def rename_variables(old_name: str, new_name: str):
    if app.data.empty:
        raise HTTPException(status_code=500, detail="No data found!")

    try:
        app.data.rename(columns={old_name: new_name}, inplace=True)

        return {"message": f"Variable '{old_name}' renamed to '{new_name}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/update_value")
async def update_value(row: int, column: str, new_value: str):
    if app.data is None or app.data.empty:
        raise HTTPException(status_code=500, detail="No data found!")

    try:
        if column not in app.data.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found in data!")
        if row < 0 or row >= len(app.data):
            raise HTTPException(status_code=400, detail=f"Index '{row}' out of range!")

        app.data.at[row, column] = new_value

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/select")
async def select_data(columns: list[str]):
    if app.data is None or app.data.empty:
        raise HTTPException(status_code=500, detail="No data found!")

    if app.data_before_selection is None:
        app.data_before_selection = app.data.copy()

    try:
        app.data = app.data[columns]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"Selection completed successfully!"}


@router.put("/unselect")
async def unselect_data():
    if app.data is None or app.data.empty or app.data_before_selection is None or app.data_before_selection.empty:
        raise HTTPException(status_code=500, detail="No data found!")

    app.data = app.data_before_selection.copy()
    app.data_before_selection = None

    return {"message": f"Unselection completed successfully!"}


@router.put("/convert")
async def convert_to_numeric():
    app.numeric_data = app.data.copy()
    for column in app.data.columns:
        if app.data[column].dtype == 'object':
            try:
                app.numeric_data[column] = pd.to_numeric(app.numeric_data[column].str.replace(',', '.'))
            except:
                app.numeric_data[column] = app.numeric_data[column].map(string_to_date_to_number)
    return {"message": f"Conversion completed successfully!"}

@router.get("/columnstats")
async def data_stats(column: str):
    if app.data is None or app.data.empty:
        raise HTTPException(status_code=500, detail="No data found!")
    
    column_data = None
    if app.numeric_data is not None:
        column_data = app.numeric_data[column]
    else:
        column_data = app.data[column]

    #descriptive
    mean = column_data.mean()
    median = column_data.median()
    dominants = column_data.mode().tolist() #there could be more than one dominant
    minimum = column_data.min()
    maximum = column_data.max()
    range = np.ptp(column_data)
    quartiles = column_data.quantile([0.25, 0.5, 0.75]).tolist()
    std_dev = column_data.std()
    variance = column_data.var()
    kurtosis_value = kurtosis(column_data)
    skewness_value = skew(column_data)

    return {
        "column_stats": {
            "mean": mean,
            "median": median,
            "dominants": dominants,
            "min": minimum,
            "max": maximum,
            "range": range,
            "quartiles": quartiles,
            "standard_deviation": std_dev,
            "variance": variance,
            "kurtosis": kurtosis_value,
            "skewness": skewness_value
        }
    }


    



@router.put("/normalize")
async def normalize_data(normalization: str):
    if app.numeric_data.empty:
        raise HTTPException(status_code=500, detail="No data found!")

    app.normalized_data = app.numeric_data.copy()

    if normalization == "minmax":
        for column in app.normalized_data.columns:
            if app.normalized_data[column].dtype != np.number:
                continue
            app.normalized_data[column] = (app.normalized_data[column] - app.normalized_data[column].min()) / (
                    app.normalized_data[column].max() - app.normalized_data[column].min())
    elif normalization == "standarization":
        for column in app.normalized_data.columns:
            if app.normalized_data[column].dtype != np.number:
                continue
            app.normalized_data[column] = (app.normalized_data[column] - app.normalized_data[column].mean()) / \
                                          app.normalized_data[column].std()
    elif normalization == "log":
        for column in app.normalized_data.columns:
            if app.normalized_data[column].dtype != np.number or app.normalized_data[column].min() <= 0:
                continue
            app.normalized_data[column] = np.log(app.normalized_data[column])
    else:
        raise HTTPException(status_code=400, detail=f"Normalization {normalization} not implemented!!")

    return {"data": list(app.normalized_data.to_dict(orient="records"))}


@router.put("/pca")
async def pca_analysis(n_components: int):
    if app.normalized_data is None or app.normalized_data.empty:
        raise HTTPException(status_code=500, detail="No data found!")

    numeric_data = app.normalized_data.select_dtypes(include=[np.number])

    if numeric_data.shape[1] < n_components:
        raise HTTPException(status_code=400,
                            detail="Number of components is greater than the number of numeric features")

    try:
        pca = PCA(n_components)
        components = pca.fit_transform(numeric_data)
        labels = {str(i): f"PC {i + 1}" for i in range(n_components)}

        result = {
            "components": components.tolist(),
            "labels": labels,
            "dimensions": [i for i in range(n_components)],
            "explained_variance": pca.explained_variance_ratio_.sum()
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


@router.get("/kMeans/visualization")
async def kMeans_visulalization():
    if app.kMeans_data is None:
        raise HTTPException(status_code=500, detail="No kMeans clustering data found!")

    numeric_data = app.normalized_data.select_dtypes(include=[np.number])
    pca = PCA(2)
    components = pca.fit_transform(numeric_data)
    x = [a[0] for a in components]
    y = [a[1] for a in components]
    fig = px.scatter(
        x=x, y=y, color=app.kMeans_data,
        title="kMeans Clustering Visualization",
        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}
    )
    png_bytes = pio.to_image(fig, format="png")
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@router.get("/kMeans/visualization2/{n_clusters}")
async def kMeans_visulalization2(n_clusters: int):
    if app.kMeans_data is None:
        raise HTTPException(status_code=500, detail="No kMeans clustering data found!")

    a = [0 for _ in range(max(app.kMeans_data) + 1)]
    for b in app.kMeans_data:
        a[b] += 1

    fig = px.bar(x=[i for i in range(max(app.kMeans_data) + 1)], y=a,
                 labels={'x': 'Number of cluster', 'y': 'No. of records in cluster'},
                 title='Number of records in each cluster')

    png_bytes = pio.to_image(fig, format="png")
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@router.get("/kMeans/{n_clusters}")
async def clustering_kMeans(n_clusters: int):
    if app.normalized_data is None or app.normalized_data.empty:
        raise HTTPException(status_code=500, detail="No data found!")

    kmeans = KMeans(n_clusters=n_clusters)
    app.cluster_data = kmeans.fit_predict(app.normalized_data)
    app.kMeans_data = app.cluster_data

    return {"message": f"Clustering 'kMeans' completed successfully!"}


@router.get("/DBSCAN/visualization")
async def DBSCAN_visulalization():
    if app.DBSCAN_data is None:
        raise HTTPException(status_code=500, detail="No DBSCAN clustering data found!")

    numeric_data = app.normalized_data.select_dtypes(include=[np.number])
    pca = PCA(2)
    components = pca.fit_transform(numeric_data)
    x = [a[0] for a in components]
    y = [a[1] for a in components]
    fig = px.scatter(
        x=x, y=y, color=app.DBSCAN_data,
        title="DBSCAN Clustering Visualization",
        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}
    )
    png_bytes = pio.to_image(fig, format="png")
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@router.get("/DBSCAN/visualization2/{n_clusters}")
async def DBSCAN_visulalization2(n_clusters: int):
    if app.DBSCAN_data is None:
        raise HTTPException(status_code=500, detail="No kMeans clustering data found!")

    a = [0 for _ in range(max(app.DBSCAN_data) + 2)]
    for b in app.DBSCAN_data:
        a[b] += 1

    fig = px.bar(x=[i for i in range(max(app.DBSCAN_data) + 2)], y=a,
                 labels={'x': 'Number of cluster', 'y': 'No. of records in cluster'},
                 title='Number of records in each cluster')

    png_bytes = pio.to_image(fig, format="png")
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@router.get("/DBSCAN/{eps}/{min_samples}")
async def clustering_DBSCAN(eps: float, min_samples: int):
    if app.normalized_data is None or app.normalized_data.empty:
        raise HTTPException(status_code=500, detail="No data found!")

    # NOT TESTED!!!
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    app.cluster_data = dbscan.fit_predict(app.normalized_data)
    app.DBSCAN_data = app.cluster_data

    return {"message": f"Clustering 'DBSCAN' completed successfully!"}


@router.get("/agglomerative/visualization")
async def agglomerative_visulalization():
    if app.cluster_data is None:
        raise HTTPException(status_code=500, detail="No agglomerative clustering data found!")

    numeric_data = app.normalized_data.select_dtypes(include=[np.number])
    pca = PCA(2)
    components = pca.fit_transform(numeric_data)
    x = [a[0] for a in components]
    y = [a[1] for a in components]
    fig = px.scatter(
        x=x, y=y, color=app.cluster_data,
        title="Agglomerative Clustering Visualization",
        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}
    )
    png_bytes = pio.to_image(fig, format="png")
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@router.get("/agglomerative/visualization2/{n_clusters}")
async def agglomerative_visulalization2(n_clusters: int):
    if app.cluster_data is None:
        raise HTTPException(status_code=500, detail="No kMeans clustering data found!")

    a = [0 for _ in range(max(app.cluster_data) + 1)]
    for b in app.cluster_data:
        a[b] += 1

    fig = px.bar(x=[i for i in range(max(app.cluster_data) + 1)], y=a,
                 labels={'x': 'Number of cluster', 'y': 'No. of records in cluster'},
                 title='Number of records in each cluster')

    png_bytes = pio.to_image(fig, format="png")
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@router.get("/agglomerative/{n_clusters}")
async def clustering_agglomerative(n_clusters: int):
    if app.normalized_data is None or app.normalized_data.empty:
        raise HTTPException(status_code=500, detail="No data found!")

    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    app.cluster_data = agglomerative.fit_predict(app.normalized_data)

    return {"message": f"Agglomerative Clustering completed successfully!"}


@router.get("/meanShift")
async def clustering_meanShift():
    if app.normalized_data is None or app.normalized_data.empty:
        raise HTTPException(status_code=500, detail="No data found!")

    mean_shift = MeanShift()
    app.cluster_data = mean_shift.fit_predict(app.normalized_data)

    return {"message": f"MeanShift Clustering completed successfully!"}


@router.get("/meanShift/visualization")
async def meanShift_visulalization():
    if app.cluster_data is None:
        raise HTTPException(status_code=500, detail="No MeanShift clustering data found!")

    numeric_data = app.normalized_data.select_dtypes(include=[np.number])
    pca = PCA(2)
    components = pca.fit_transform(numeric_data)
    x = [a[0] for a in components]
    y = [a[1] for a in components]
    fig = px.scatter(
        x=x, y=y, color=app.cluster_data,
        title="MeanShift Clustering Visualization",
        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}
    )
    png_bytes = pio.to_image(fig, format="png")
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@router.get("/meanShift/visualization2/{n_clusters}")
async def meanShift_visulalization2(n_clusters: int):
    if app.cluster_data is None:
        raise HTTPException(status_code=500, detail="No meanShift clustering data found!")

    a = [0 for _ in range(max(app.cluster_data) + 1)]
    for b in app.cluster_data:
        a[b] += 1

    fig = px.bar(x=[i for i in range(max(app.cluster_data) + 1)], y=a,
                 labels={'x': 'Number of cluster', 'y': 'No. of records in cluster'},
                 title='Number of records in each cluster')

    png_bytes = pio.to_image(fig, format="png")
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@router.get("/affinity")
async def clustering_affinity():
    if app.normalized_data is None or app.normalized_data.empty:
        raise HTTPException(status_code=500, detail="No data found!")

    affinity = AffinityPropagation()
    app.cluster_data = affinity.fit_predict(app.normalized_data)

    return {"message": f"Affinity Clustering completed successfully!"}


@router.get("/affinity/visualization")
async def affinity_visulalization():
    if app.cluster_data is None:
        raise HTTPException(status_code=500, detail="No Affinity clustering data found!")

    numeric_data = app.normalized_data.select_dtypes(include=[np.number])
    pca = PCA(2)
    components = pca.fit_transform(numeric_data)
    x = [a[0] for a in components]
    y = [a[1] for a in components]
    fig = px.scatter(
        x=x, y=y, color=app.cluster_data,
        title="Affinity Clustering Visualization",
        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}
    )
    png_bytes = pio.to_image(fig, format="png")
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@router.get("/affinity/visualization2/{n_clusters}")
async def affinity_visulalization2(n_clusters: int):
    if app.cluster_data is None:
        raise HTTPException(status_code=500, detail="No Affinity clustering data found!")

    a = [0 for _ in range(max(app.cluster_data) + 1)]
    for b in app.cluster_data:
        a[b] += 1

    fig = px.bar(x=[i for i in range(max(app.cluster_data) + 1)], y=a,
                 labels={'x': 'Number of cluster', 'y': 'No. of records in cluster'},
                 title='Number of records in each cluster')

    png_bytes = pio.to_image(fig, format="png")
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


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


@router.get("/get_nclusters")
async def get_nclusters():
    return {"nclusters": f"{max(app.cluster_data) + 1}"}


# include router
app.include_router(router)
