import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

with open("model/insurance_cross_sell_model_v.0.1.pkl", 'rb') as file:
    model = pickle.load(file)
    
class CustomerData(BaseModel):
    Gender: int
    Age: int
    Driving_License: int
    Previously_Insured: int
    Vehicle_Age: int
    Vehicle_Damage: int

@app.post("/predict/")
async def predict(data: CustomerData):
    input_data = [[
        int(data.Gender), int(data.Age), int(data.Driving_License),
        int(data.Previously_Insured), int(data.Vehicle_Age), int(data.Vehicle_Damage)
    ]]
    
    prediction = model.predict(input_data)
    prediction_python = prediction[0].item() if isinstance(prediction[0], np.generic) else prediction[0]
    if prediction_python == 0:
        prediction_converted = "Customer is not interested"
    else:
        prediction_converted = "Customer is interested"
    
    return {"prediction": prediction_converted}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
