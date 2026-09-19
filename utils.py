import numpy as np
import matplotlib.pyplot as plt

#Helper function to inverse-transform 1D predicted power arrays.
def inverse_transform_helper(pred_col, scaler):
    pred_col = pred_col.reshape(-1, 1)
    dummy = np.zeros((len(pred_col), 3)) 
    dummy[:, 2] = pred_col.flatten()    
    inverse_data = scaler.inverse_transform(dummy)
    return inverse_data[:, 2] 

#Calculate RMSE and MAE metrics.
def calculate_metrics(real_pred, real_true):
    rmse = np.sqrt(np.mean((real_pred - real_true)**2))
    mae = np.mean(np.abs(real_pred - real_true))
    return rmse, mae

#Generate and save comparison plots between actual and predicted values
def plot_predictions(real_true, real_pred, top_k, rmse, cut_in_threshold, plot_len=300):   .
    plt.figure(figsize=(18, 6))
    plt.plot(range(plot_len), real_true[:plot_len], label='Actual Power', color='black', alpha=0.6)
    plt.plot(range(plot_len), real_pred[:plot_len], label='Predicted Power (Cut-in Rule Applied)', color='red', linestyle='--')
    plt.title(f"GRA-LLM Predictions with Physical Constraint (<{cut_in_threshold}m/s -> 0 MW) | RMSE: {rmse:.2f}")
    plt.xlabel("Time Steps")
    plt.ylabel("Active Power (kW)")
    plt.legend()
    
    save_path = f'final_cutin_top{top_k}.png'
    plt.savefig(save_path)
    print(f"[INFO] Plot saved successfully to: {save_path}")