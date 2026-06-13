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

    # Cek apakah MLflow sudah buat run otomatis (karena MLproject atau backend lain)
    # Kalau sudah ada, pakai itu. Kalau belum, buat baru.
    active_run = mlflow.active_run()
    
    if active_run:
        print(f"Menggunakan run aktif yang sudah ada: {active_run.info.run_id}")
        run = active_run
    else:
        print("Memulai run baru...")
        run = mlflow.start_run()

    with run:
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        mlflow.sklearn.log_model(
            sk_model=rf, 
            artifact_path="model",
            pip_requirements="requirements.txt" 
        )
        
        # Simpan ID ke file agar Docker bisa baca
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
            
        print(f"Selesai! Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_model()