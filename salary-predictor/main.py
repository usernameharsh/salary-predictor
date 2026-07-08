from fastapi import FastAPI
from pydantic import BaseModel
import joblib


model = joblib.load("salarypredictor.pkl")
preprocessor = joblib.load("preprocessor.pkl")

encoders = preprocessor['encoders']
top_10_title = preprocessor['top_10_title']

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}


class SalaryInput(BaseModel):
    work_year: int            
    experience_level: str      
    employment_type: str       
    job_title: str             
    employee_residence: str   
    remote_ratio: int          
    company_location: str      
    company_size: str          

@app.post("/predict")
def predict(data: SalaryInput):
    
    job = data.job_title if data.job_title in top_10_title else 'Other'
    
    
    def safe_encode(col_name, value):
        le = encoders[col_name]
        
        if value in le.classes_:
            return int(le.transform([value])[0])
        else:
            return int(le.transform([le.classes_[0]])[0])

    
    encoded_exp = safe_encode('experience_level', data.experience_level)
    encoded_emp = safe_encode('employment_type', data.employment_type)
    encoded_job = safe_encode('job_title', job)
    encoded_res = safe_encode('employee_residence', data.employee_residence)
    encoded_loc = safe_encode('company_location', data.company_location)
    encoded_size = safe_encode('company_size', data.company_size)

    
    features = [[
        data.work_year,
        encoded_exp,
        encoded_emp,
        encoded_job,
        encoded_res,
        data.remote_ratio,
        encoded_loc,
        encoded_size
    ]]
    
    
    prediction = model.predict(features)[0]
    return {"predicted_salary_usd": round(float(prediction), 2)}