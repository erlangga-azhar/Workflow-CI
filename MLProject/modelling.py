import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model():
    print("Memuat dataset untuk CI...")
    df = pd.read_csv("titanic_clean.csv")
    X = df.drop(columns=['survived'])
    y = df['survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tracking ke folder lokal (mlruns)
    mlflow.set_tracking_uri("local")
    
    with mlflow.start_run() as run:
        print("Melatih model...")
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        
        # Wajib log model dengan nama "model"
        mlflow.sklearn.log_model(rf, "model")
        
        # Menyimpan Run ID untuk GitHub Actions
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
            
        print(f"Selesai! Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_model()