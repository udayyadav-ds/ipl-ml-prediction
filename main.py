from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="IPL Match Predictor API")

# Load model and artifacts
try:
    with open('ipl_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    model = artifacts['model']
    le_team = artifacts['le_team']
    le_venue = artifacts['le_venue']
    le_toss = artifacts['le_toss']
except Exception as e:
    print(f"Error loading model: {e}")

class MatchInput(BaseModel):
    team1: str
    team2: str
    venue: str
    toss_winner: str
    toss_decision: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the IPL Match Predictor API"}

@app.post("/predict")
def predict_winner(match: MatchInput):
    try:
        # Encode inputs
        t1_encoded = le_team.transform([match.team1])[0]
        t2_encoded = le_team.transform([match.team2])[0]
        v_encoded = le_venue.transform([match.venue])[0]
        tw_encoded = le_team.transform([match.toss_winner])[0]
        td_encoded = le_toss.transform([match.toss_decision])[0]
        
        # Prepare feature vector
        features = np.array([[t1_encoded, t2_encoded, v_encoded, tw_encoded, td_encoded]])
        
        # Predict
        prediction_encoded = model.predict(features)[0]
        prediction_team = le_team.inverse_transform([prediction_encoded])[0]
        
        # Probability
        probabilities = model.predict_proba(features)[0]
        prob_dict = {le_team.inverse_transform([i])[0]: float(p) for i, p in enumerate(probabilities) if p > 0}
        
        return {
            "predicted_winner": prediction_team,
            "win_probability": prob_dict.get(prediction_team, 0.0),
            "all_probabilities": prob_dict
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metadata")
def get_metadata():
    return {
        "teams": list(le_team.classes_),
        "venues": list(le_venue.classes_),
        "toss_decisions": list(le_toss.classes_)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
