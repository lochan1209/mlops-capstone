import pandas.core.dtypes.common as _pd_common

if not hasattr(_pd_common, "infer_dtype_from_object"):
    import pandas.api.types as _pat
    _pd_common.infer_dtype_from_object = _pat.infer_dtype

import os, shutil

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"   # 3.17 refuses a local file store without this
shutil.rmtree("mlruns", ignore_errors=True)      # the capstone starts from an empty store

import mlflow
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("student-performance")
NAME = "student-performance-rf"

URL = "https://raw.githubusercontent.com/krishnaik06/mlproject/main/notebook/data/stud.csv"
df = pd.read_csv(URL)
train, test = train_test_split(df, test_size=0.2, random_state=42)
Xtr, Xte = train.drop(columns=["math_score"]), test.drop(columns=["math_score"])

# the preprocessor is fitted on the training rows only, so the test rows stay unseen
pre = ColumnTransformer([
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
     Xtr.select_dtypes(include="object").columns.tolist()),
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]),
     Xtr.select_dtypes(exclude="object").columns.tolist()),
])

model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)

# the fit happens inside a tracked run, so the params, the score and the model are recorded
with mlflow.start_run(run_name="capstone"):
    mlflow.log_params({"n_estimators": 100, "max_depth": 12, "random_state": 42})
    model.fit(pre.fit_transform(Xtr), train.math_score)
    r2 = round(r2_score(test.math_score, model.predict(pre.transform(Xte))), 4)
    mlflow.log_metrics({"r2": r2})
    info = mlflow.sklearn.log_model(model, name="model",
                                    skops_trusted_types=["sklearn.tree._tree.Tree"])
print("trained      r2", r2)

# the registry and the alias: the version other code asks for by name, not by run id
version = mlflow.register_model(info.model_uri, NAME).version
MlflowClient().set_registered_model_alias(NAME, "champion", version)
champion = mlflow.sklearn.load_model(f"models:/{NAME}@champion")
print("registered   version", version)
print("champion     r2", round(r2_score(test.math_score, champion.predict(pre.transform(Xte))), 4))

# Export both the fitted preprocessor and the champion model
joblib.dump(pre, "preprocessor.joblib")
joblib.dump(champion, "model.joblib")

class StudentRecord(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int = Field(ge=0, le=100)
    writing_score: int = Field(ge=0, le=100)


api = FastAPI(title="Student Performance API")


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/predict")
def predict(record: StudentRecord):
    row = pd.DataFrame([record.model_dump()])
    return {"math_score": round(float(model.predict(pre.transform(row))[0]), 2)}


client = TestClient(api)
RECORD = {"gender": "female", "race_ethnicity": "group B",
          "parental_level_of_education": "bachelor's degree", "lunch": "standard",
          "test_preparation_course": "none", "reading_score": 72, "writing_score": 74}
print("health      ", client.get("/health").json())
print("predict     ", client.post("/predict", json=RECORD).json())

# the failure mode: a score outside the range the schema allows never reaches the model
bad = client.post("/predict", json={**RECORD, "reading_score": 150})
print("bad request ", bad.status_code, bad.json()["detail"][0]["msg"])

# the drift check reads only what a request carries, so math_score, the target, is not in it
COLS = ["reading_score", "writing_score", "gender", "lunch"]
definition = DataDefinition(numerical_columns=COLS[:2], categorical_columns=COLS[2:])
reference = Dataset.from_pandas(df[df.test_preparation_course == "none"][COLS],
                                data_definition=definition)
current = Dataset.from_pandas(df[df.test_preparation_course == "completed"][COLS],
                              data_definition=definition)
report = Report(metrics=[DataDriftPreset()]).run(current_data=current, reference_data=reference)
print("drift       ", report.dict()["metrics"][0]["value"])