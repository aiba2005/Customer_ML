from fastapi import FastAPI
import joblib
import uvicorn
from pydantic import BaseModel

customer_app = FastAPI()

scaler = joblib.load('scaler_2.pkl')
model_lg = joblib.load('model_lg (1).pkl')
model_svm = joblib.load('model_svm.pkl')


contract = ['Month-to-month', 'One year', 'Two year']
internetService = ['DSL', 'Fiber optic', 'No']
onlineSecurity =['No', 'No internet service', 'Yes']
techSupport = ['No', 'No internet service', 'Yes']










class CustomerChema(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    OnlineSecurity: str
    TechSupport: str




@customer_app.post('/predict/log/')
async def predict(cust: CustomerChema):
    cust_dict = cust.dict()

    Contract_1 = cust_dict.pop('Contract')
    Contract0_1 = [ 1 if Contract_1 == i else 0 for i in contract]



    InternetService1 = cust_dict.pop('InternetService')
    InternetService0_1 = [1 if InternetService1 == i else 0 for i in internetService]


    OnlineSecurity1 = cust_dict.pop('OnlineSecurity')
    OnlineSecurity0_1 = [1 if OnlineSecurity1 == i else 0 for i in onlineSecurity]


    TechSupport1 = cust_dict.pop('TechSupport')
    TechSupport0_1 = [1 if TechSupport1 == i else 0 for i in techSupport]


    features = list(cust_dict.values()) + Contract0_1 + InternetService0_1 + OnlineSecurity0_1 + TechSupport0_1

    scaled_data = scaler.transform([features])
    pred = model_lg.predict(scaled_data)[0]
    return {"prediction": "Yes" if pred == 1 else "No"}



@customer_app.post('/predict/svm/')
async def predict(cust: CustomerChema):
    cust_dict = cust.dict()

    Contract_1 = cust_dict.pop('Contract')
    Contract0_1 = [ 1 if Contract_1 == i else 0 for i in contract]



    InternetService1 = cust_dict.pop('InternetService')
    InternetService0_1 = [1 if InternetService1 == i else 0 for i in internetService]


    OnlineSecurity1 = cust_dict.pop('OnlineSecurity')
    OnlineSecurity0_1 = [1 if OnlineSecurity1 == i else 0 for i in onlineSecurity]


    TechSupport1 = cust_dict.pop('TechSupport')
    TechSupport0_1 = [1 if TechSupport1 == i else 0 for i in techSupport]


    features = list(cust_dict.values()) + Contract0_1 + InternetService0_1 + OnlineSecurity0_1 + TechSupport0_1

    scaled_data = scaler.transform([features])
    pred = model_svm.predict(scaled_data)[0]
    return {"prediction": "Yes" if pred == 1 else "No"}





if __name__ == '__main__':
    uvicorn.run(customer_app, host='127.0.0.1', port=7788)
