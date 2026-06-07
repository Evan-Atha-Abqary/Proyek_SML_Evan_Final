import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def run_training():
    print("Mulai membaca data bersih...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    clean_path = os.path.join(current_dir, '..', 'preprocessing', 'credit_risk_dataset_clean.csv')
    df = pd.read_csv(clean_path)
    
    print("Membagi data untuk pelatihan...")
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- PENANGKAL ERROR WINDOWS ---
    # Memaksa MLflow membuat folder mlruns di lokasi yang benar tanpa membaca spasi
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Eksperimen_Prediksi_Kredit_Evan")
    # -------------------------------
    
    with mlflow.start_run():
        print("Melatih model Random Forest (AI sedang belajar)...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        print("Menguji model...")
        y_pred = model.predict(X_test)
        akurasi = accuracy_score(y_test, y_pred)
        
        print(f"Hasil Akurasi Model: {akurasi * 100:.2f}%")
        
        print("Menyimpan rekam jejak ke MLflow...")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_metric("accuracy", akurasi)
        mlflow.sklearn.log_model(model, "model_random_forest")
        
    print("Selesai! Model berhasil dilatih dan direkam oleh MLflow.")

if __name__ == "__main__":
    run_training()