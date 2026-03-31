import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_experiment("IPL_Prediction_Model")

def train_model():

    df = pd.read_csv("processed_matches.csv")

    features = [
        'team1_encoded',
        'team2_encoded',
        'venue_encoded',
        'team1_win_ratio',
        'team2_win_ratio',
        'venue_avg_runs'
    ]

    X = df[features]
    y = df['winner_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():

        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("learning_rate", 0.1)

        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, "model")

        print("Accuracy:", acc)

        with open("ipl_model.pkl", "wb") as f:
            pickle.dump(model, f)


if __name__ == "__main__":
    train_model()