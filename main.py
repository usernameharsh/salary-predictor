
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os
import joblib
model = joblib.load("salarypredictor.pkl")
app = FastAPI()
@app.get("/health")
def health():
    return {"status": "ok"}
class SalaryInput(BaseModel):
    work_year: int
    experience_level: int
    employment_type: int
    job_title: int
    employee_residence: int
    remote_ratio: int
    company_location: int
    company_size: int


@app.post("/predict")
def predict(data: SalaryInput):
    features = [[
        data.work_year,
        data.experience_level,
        data.employment_type,
        data.job_title,
        data.employee_residence,
        data.remote_ratio,
        data.company_location,
        data.company_size
    ]]
    prediction = model.predict(features)[0]
    return {"predicted_salary_usd": round(float(prediction), 2)}