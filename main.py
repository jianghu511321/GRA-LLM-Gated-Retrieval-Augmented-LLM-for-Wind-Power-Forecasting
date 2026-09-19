import torch
import torch.nn as nn
import joblib
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from config import *
from dataset import GatedRAFDataset
from model import AttentionRAF_GPT2
from utils import inverse_transform_helper, calculate_metrics, plot_predictions

def main():
    print(f"[GRA-LLM Framework Pipeline] Starting... | Device: {DEVICE}")

    # 1. Load Preprocessed Data
    try:
        X_train = joblib.load(os.path.join(DATA_DIR, 'train_set_pro.pkl'))
        Y_train = joblib.load(os.path.join(DATA_DIR, 'train_label_pro.pkl'))
        KB_X    = joblib.load(os.path.join(DATA_DIR, 'kb_set_x_pro.pkl')) 
        KB_Y    = joblib.load(os.path.join(DATA_DIR, 'kb_set_y_pro.pkl')) 
        X_test  = joblib.load(os.path.join(DATA_DIR, 'test_set_pro.pkl'))
        Y_test  = joblib.load(os.path.join(DATA_DIR, 'test_label_pro.pkl'))
        scaler  = joblib.load(os.path.join(DATA_DIR, 'scaler.pkl'))
    except FileNotFoundError:
        print("[ERROR] Preprocessed data not found. Please run 'data_preprocessing.py' first.")
        return

    print(f"[INFO] Train Samples: {X_train.shape[0]}, Test Samples: {X_test.shape[0]}")

    train_dataset = GatedRAFDataset(X_train, Y_train, kb_x=KB_X, kb_y=KB_Y, k=TOP_K)
    test_dataset  = GatedRAFDataset(X_test, Y_test, kb_x=KB_X, kb_y=KB_Y, k=TOP_K)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AttentionRAF_GPT2(original_dim=INPUT_DIM, k=TOP_K, output_dim=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.SmoothL1Loss() 

    # 2. Training Phase
    print(f"\n>>> Starting Training Phase...")
    best_loss = float('inf')
    save_path = f'best_gra_llm_top{TOP_K}.pth'

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            q_x = batch['query_x'].to(DEVICE)
            r_x = batch['retrieved_x'].to(DEVICE)
            r_y = batch['retrieved_y'].to(DEVICE)
            y_true = batch['target'].to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(q_x, r_x, r_y)
            loss = criterion(preds, y_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            if (epoch+1) % 5 == 0: 
                print(f"Epoch {epoch+1:03d}/{EPOCHS} | Loss: {avg_loss:.6f} [New Best Model Saved]")
        elif (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:03d}/{EPOCHS} | Loss: {avg_loss:.6f}")

    # 3. Testing & Inference Phase
    print("\n>>> Starting Testing Phase (Applying Cut-In Wind Speed Physical Rule)...")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    predictions, ground_truth, wind_speeds_norm = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            q_x = batch['query_x'].to(DEVICE)
            r_x = batch['retrieved_x'].to(DEVICE)
            r_y = batch['retrieved_y'].to(DEVICE)
            
            p = model(q_x, r_x, r_y)
            
            predictions.extend(p.cpu().numpy())
            ground_truth.extend(batch['target'].cpu().numpy())
            # Extract the wind speed at the last time step for physical rule checking
            wind_speeds_norm.extend(q_x[:, -1, 0].cpu().numpy()) 

    # 4. Post-Processing & Physical Constraint
    pred_np = np.clip(np.array(predictions), 0, 1.05) 
    true_np = np.array(ground_truth)
    ws_norm_np = np.array(wind_speeds_norm)
    
    real_pred = inverse_transform_helper(pred_np, scaler)
    real_true = inverse_transform_helper(true_np, scaler)
    
    # Reverse normalize actual wind speeds: X = (X_norm - min) / scale
    real_ws = (ws_norm_np - scaler.min_[0]) / scaler.scale_[0] 
    
    # Physical Constraint: Cut-in Threshold
    cut_in_mask = real_ws < CUT_IN_SPEED_THRESHOLD
    print(f"[CONSTRAINT] Detected {np.sum(cut_in_mask)} samples where wind speed < {CUT_IN_SPEED_THRESHOLD} m/s. Forcing predictions to 0 MW.")
    
    real_pred[cut_in_mask] = 0.0
    real_pred = np.maximum(real_pred, 0) # Ensure strictly non-negative predictions

    # 5. Metrics Evaluation
    rmse, mae = calculate_metrics(real_pred, real_true)
    print("\n" + "=" * 40)
    print(f" Final Evaluation Results (Cut-In Applied):")
    print(f" RMSE : {rmse:.4f} kW")
    print(f" MAE  : {mae:.4f} kW")
    print("=" * 40 + "\n")

    # 6. Save Results & Plotting
    df_res = pd.DataFrame({'Actual_Power': real_true, 'Predicted_Power': real_pred, 'Actual_WindSpeed': real_ws})
    df_res.to_csv(f'result_gra_llm_top{TOP_K}.csv', index=False)
    
    plot_predictions(real_true, real_pred, TOP_K, rmse, CUT_IN_SPEED_THRESHOLD)
    print("Pipeline executed successfully")

if __name__ == "__main__":
    main()