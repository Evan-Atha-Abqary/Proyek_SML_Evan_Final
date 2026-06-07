import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def run_inference():
    # 1. MENCARI ALAMAT LENGKAP FOLDER MLRUNS
    # Ambil lokasi file ini (modeling/predict.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Naik satu folder ke folder utama (root)
    root_dir = os.path.dirname(current_dir)
    # Tentukan path mutlak ke mlruns
    mlruns_path = os.path.join(root_dir, 'mlruns')
    
    # 2. Set Tracking URI dengan format yang disukai Windows
    mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
    
    print(f"Mencari model di folder: {mlruns_path}")
    
    # --- MASUKKAN RUN_ID KAMU DI SINI ---
    RUN_ID = '5c3d3906b2f3459fb1ec0935d648cc7a'
    # ------------------------------------
    
    print(f"Menggunakan Run ID: {RUN_ID}")
    
    # 3. Load model
    model_uri = f"runs:/{RUN_ID}/model_random_forest"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Gagal memuat model! Pastikan Run ID benar. Error: {e}")
        return
    
    print("Membaca data untuk evaluasi...")
    clean_path = os.path.join(root_dir, 'preprocessing', 'credit_risk_dataset_clean.csv')
    df = pd.read_csv(clean_path)
    
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    print("Melakukan Prediksi dan Evaluasi...")
    y_pred = model.predict(X)
    
    print("\n--- Hasil Evaluasi (Classification Report) ---")
    print(classification_report(y, y_pred))
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y, y_pred))

if __name__ == "__main__":
    run_inference()