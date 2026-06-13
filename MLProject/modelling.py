import pandas as pd
import mlflow
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model():
    print("Memuat dataset untuk CI...")
    df = pd.read_csv("titanic_clean.csv")
    X = df.drop(columns=['survived'])
    y = df['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # FIX: Cek apakah ada run aktif, kalau tidak ada baru start
    run = mlflow.active_run()
    if run is None:
        run = mlflow.start_run()
        
    with run:
        print(f"Melatih model pada run: {run.info.run_id}")
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(rf, "model")
        
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
            
        print(f"Selesai! Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_model()