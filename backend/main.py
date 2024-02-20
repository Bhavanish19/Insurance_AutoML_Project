import io
import pandas as pd
import h2o
from fastapi import FastAPI, File
from fastapi.responses import JSONResponse, HTMLResponse
import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from utils.data_processing import match_col_types, separate_id_col

app = FastAPI()
h2o.init()
client = MlflowClient()

# Get the best model based on log_loss metric
all_exps = [exp.experiment_id for exp in client.list_experiments()]
runs = mlflow.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ALL)
best_run = runs.loc[runs['metrics.log_loss'].idxmin()]
run_id, exp_id = best_run['run_id'], best_run['experiment_id']
print(f'Loading best model: Run {run_id} of Experiment {exp_id}')
best_model = mlflow.h2o.load_model(f"mlruns/{exp_id}/{run_id}/artifacts/model/")

@app.post("/predict")
async def predict(file: bytes = File(...)):
    print('[+] Initiate Prediction')
    test_df = pd.read_csv(io.BytesIO(file))
    test_h2o = h2o.H2OFrame(test_df)
    id_name, X_id, X_h2o = separate_id_col(test_h2o)
    X_h2o = match_col_types(X_h2o)
    preds = best_model.predict(X_h2o)
    preds_final = preds.as_data_frame()['predict'].tolist()
    if id_name is not None:
        id_list = X_id.as_data_frame()[id_name].tolist()
        preds_final = dict(zip(id_list, preds_final))
    return JSONResponse(content=preds_final)

@app.get("/")
async def main():
    content = """
    <body>
    <h2>Welcome to the End-to-End AutoML Pipeline Project for Insurance. </h2>
    <p>The H2O model and FastAPI instances have been set up successfully.</p>
    <p>You can view the FastAPI UI by heading to localhost:8000.</p>
    <p>Proceed to initialize the Streamlit UI (frontend/app.py) to submit prediction requests.</p>
    </body>
    """
    return HTMLResponse(content=content)
