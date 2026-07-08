import gradio as grd
import joblib

model = joblib.load('salarypredictor.pkl')
preprocess = joblib.load('preprocessor.pkl')

encoders = preprocess['encoders']
top_10_title = preprocess['top_10_title']


def pred(work_year, experience_level, employment_type, 
            job_title, employee_residence, remote_ratio, 
            company_location, company_size):
    
    job = job_title if job_title in top_10_title else 'Other'
    
    
    def safe_encode(col_name, value):
        le = encoders[col_name]
        
        if value in le.classes_:
            return int(le.transform([value])[0])
        else:
            return int(le.transform([le.classes_[0]])[0])

    
    encoded_exp = safe_encode('experience_level',experience_level)
    encoded_emp = safe_encode('employment_type',employment_type)
    encoded_job = safe_encode('job_title', job)
    encoded_res = safe_encode('employee_residence', employee_residence)
    encoded_loc = safe_encode('company_location',company_location)
    encoded_size = safe_encode('company_size', company_size)

    
    features = [[
        work_year,
        encoded_exp,
        encoded_emp,
        encoded_job,
        encoded_res,
        remote_ratio,
        encoded_loc,
        encoded_size
    ]]
    prediction = model.predict(features)[0]
    return f"Predicted Salary: ${round(float(prediction), 2):,}"

demo = grd.Interface(
    fn = pred,
    inputs = [
        grd.Dropdown([2025,2024,2023,2022,2021,2020], label = "Work Year"),
        grd.Dropdown(["EN","MI","SE","EX"], label = "experience_level"),
        grd.Dropdown(["FT","CT","PT","FL"], label="Employment Type"),
        grd.Textbox(label="Job Title"),
        grd.Textbox(label="Employee Residence"),
        grd.Dropdown([0,50,100], label="Remote Ratio"),
        grd.Textbox(label="Company Location"),
        grd.Dropdown(["S","M","L"], label="Company Size")
    ],
    outputs=grd.Textbox(label="Predicted Salary (USD)")
)
demo.launch()