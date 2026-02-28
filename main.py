from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="MeniSense ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load all saved models ──
with open("model_diagnosis.pkl", "rb") as f:
    model_diag = pickle.load(f)

with open("model_stage.pkl", "rb") as f:
    model_stage = pickle.load(f)

with open("scaler_diagnosis.pkl", "rb") as f:
    scaler1 = pickle.load(f)

with open("scaler_stage.pkl", "rb") as f:
    scaler2 = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_cols = pickle.load(f)

print("✅ Models loaded. Feature columns are:")
print(feature_cols)

# ── Patient input — using YOUR real column names ──
class PatientInput(BaseModel):
    Age: float
    Gender: str
    Age_Group: str
    Vaccination_Status: str
    Comorbidities: str
    Previous_Meningitis_History: str
    Fever: float
    Headache: float
    Neck_Stiffness: float
    Photophobia: float
    Altered_Mental_Status: str
    Seizures: str
    Petechiae: str
    CSF_Pressure: float
    CSF_WBC_Count: float
    CSF_Protein: float
    CSF_Glucose: float
    Blood_WBC_Count: float
    CRP_Level: float
    Procalcitonin: float
    GCS_Score: float
    CSF_to_Blood_Glucose_Ratio: float
    CSF_Neutrophils_pct: float
    CSF_Lymphocytes_pct: float
    CSF_Culture_Result: str

def get_risk(stage):
    return {"Stage I": "Low", "Stage II": "Moderate", "Stage III": "High"}.get(stage, "Low")

def get_message(diagnosis, stage, risk):
    if diagnosis == "Normal":
        return "No meningitis indicators detected. Monitor symptoms and consult a doctor if condition changes."
    elif risk == "Low":
        return f"Early stage {diagnosis} meningitis detected. Please consult a doctor within 48 hours."
    elif risk == "Moderate":
        return f"Moderate risk {diagnosis} meningitis detected. Visit a hospital within 24 hours."
    else:
        return f"HIGH RISK — Severe {diagnosis} meningitis detected. Seek emergency care immediately."

@app.post("/predict")
def predict(data: PatientInput):
    row = data.dict()

    # Rename pct fields back to % for the model
    row["CSF_Neutrophils_%"] = row.pop("CSF_Neutrophils_pct")
    row["CSF_Lymphocytes_%"] = row.pop("CSF_Lymphocytes_pct")

    df_in = pd.DataFrame([row])

    categorical_cols = ['Gender', 'Vaccination_Status', 'Age_Group',
                        'Comorbidities', 'Previous_Meningitis_History',
                        'Petechiae', 'Seizures', 'Altered_Mental_Status',
                        'CSF_Culture_Result']
    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            try:
                df_in[col] = le.transform(df_in[col])
            except:
                df_in[col] = 0

    df_in = df_in[feature_cols]

    X1_scaled = scaler1.transform(df_in)
    diag_pred = model_diag.predict(X1_scaled)[0]
    diag_proba = model_diag.predict_proba(X1_scaled)[0]
    diag_confidence = round(float(max(diag_proba)) * 100, 1)

    stage_pred = None
    risk_level = "Low"
    if diag_pred != "Normal":
        X2_scaled = scaler2.transform(df_in)
        stage_pred = model_stage.predict(X2_scaled)[0]
        risk_level = get_risk(stage_pred)

    return {
        "diagnosis": diag_pred,
        "confidence": diag_confidence,
        "stage": stage_pred,
        "risk_level": risk_level,
        "message": get_message(diag_pred, stage_pred, risk_level)
    }

@app.get("/")
def root():
    return {"status": "MeniSense ML API is running ✅"}