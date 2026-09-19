import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import os
from config import *

def create_sequences(data_array, start_idx, end_idx, seq_len):
    xs, ys = [], []
    if end_idx > len(data_array): 
        end_idx = len(data_array)
    
    for i in range(start_idx, end_idx - seq_len):
        x = data_array[i : i+seq_len]
        y = data_array[i+seq_len, 2] # Active power is at index 2 (the 3rd column)
        xs.append(x)
        ys.append(y)
        
    return np.array(xs), np.array(ys)

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print(">>> Loading dataset...")
    try:
        df = pd.read_csv(CSV_PATH)
        data = df.iloc[:, COL_INDICES].astype(float).values
        data = pd.DataFrame(data).interpolate(limit_direction='both').values
        print(f"[SUCCESS] Dataset loaded: {data.shape[0]} rows in total.")
    except Exception as e:
        print(f"[ERROR] Failed to read dataset: {e}\nPlease ensure {CSV_PATH} exists.")
        return

    print(">>> Normalizing data...")
    scaler = MinMaxScaler()
    
    # Prevent data leakage
    train_part = data[TRAIN_START:TRAIN_END]       
    kb_part    = data[KB_START:]       
    
    fit_data = np.concatenate([train_part, kb_part], axis=0)
    scaler.fit(fit_data)
    
    # Transform all data using the fitted scaler
    data_norm = scaler.transform(data)
    
    # Save the scaler for post-processing and inference
    joblib.dump(scaler, os.path.join(DATA_DIR, 'scaler.pkl'))

    print(f"1. Building Training Set (Rows {TRAIN_START} ~ {TRAIN_END})...")
    X_train, Y_train = create_sequences(data_norm, TRAIN_START, TRAIN_END, SEQ_LEN)
    
    print(f"2. Building Testing Set (Rows {TEST_START} ~ {TEST_END})...")
    X_test, Y_test = create_sequences(data_norm, TEST_START, TEST_END, SEQ_LEN)
    
    print(f"3. Building Knowledge Base (Rows {KB_START} to End)...")
    X_kb, Y_kb = create_sequences(data_norm, KB_START, len(data_norm), SEQ_LEN)
    
    print(">>> Saving preprocessed files...")
    joblib.dump(X_train, os.path.join(DATA_DIR, 'train_set_pro.pkl'))
    joblib.dump(Y_train, os.path.join(DATA_DIR, 'train_label_pro.pkl'))
    joblib.dump(X_kb,    os.path.join(DATA_DIR, 'kb_set_x_pro.pkl'))
    joblib.dump(Y_kb,    os.path.join(DATA_DIR, 'kb_set_y_pro.pkl'))
    joblib.dump(X_test,  os.path.join(DATA_DIR, 'test_set_pro.pkl'))
    joblib.dump(Y_test,  os.path.join(DATA_DIR, 'test_label_pro.pkl'))

    print("\n" + "-" * 40)
    print("[DONE] Data preprocessing completed!")
    print(f"Train Set      -> X: {X_train.shape} | Y: {Y_train.shape}")
    print(f"Test Set       -> X: {X_test.shape}  | Y: {Y_test.shape}")
    print(f"Knowledge Base -> X: {X_kb.shape}    | Y: {Y_kb.shape}")
    print("-" * 40)

if __name__ == "__main__":
    main()