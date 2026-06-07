import pandas as pd
import os

def run_preprocessing():
    print("Mulai membaca data mentah...")
    
    # --- PERBAIKAN JALUR (PATH) DI SINI ---
    # 1. Mengambil lokasi folder persis tempat script automate_Evan.py ini berada
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Menggabungkan jalur secara absolut ke dataset_raw
    raw_path = os.path.join(current_dir, '..', 'dataset_raw', 'credit_risk_dataset.csv')
    # --------------------------------------
    
    df = pd.read_csv(raw_path)
    
    print("Mulai membersihkan data...")
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    df.drop_duplicates(inplace=True)
    df_clean = pd.get_dummies(df, drop_first=True)
    
    print("Menyimpan data bersih...")
    # Menyimpan file output di folder yang sama dengan script ini
    clean_path = os.path.join(current_dir, 'credit_risk_dataset_clean.csv')
    df_clean.to_csv(clean_path, index=False)
    
    print(f"Selesai! Data bersih berhasil diperbarui dan disimpan di: {clean_path}")
    return df_clean

if __name__ == "__main__":
    run_preprocessing()