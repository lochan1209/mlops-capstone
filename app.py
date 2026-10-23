import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Initialize FastAPI
app = FastAPI(title="Student Performance API")

# Load artifacts
pre = joblib.load("preprocessor.joblib")
model = joblib.load("model.joblib")


class StudentRecord(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int = Field(ge=0, le=100)
    writing_score: int = Field(ge=0, le=100)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(record: StudentRecord):
    row = pd.DataFrame([record.model_dump()])
    transformed = pre.transform(row)
    pred = model.predict(transformed)[0]
    return {"math_score": round(float(pred), 2)}