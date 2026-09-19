import os
import torch

# 1. Path Configuration
# Recommended folder structure: place 'wind_power_data.csv' inside the './data' directory
DATA_DIR = './data'
CSV_PATH = os.path.join(DATA_DIR, 'wind_power_data.csv')

# 2. Data Processing Parameters
# Column indices in the CSV: [Wind Speed, Wind Direction, Active Power]
COL_INDICES = [3, 4, 12] 
SEQ_LEN = 96

# Data split strategy
TRAIN_START = 0
TRAIN_END   = 30000
TEST_START  = 30000
TEST_END    = 35281
KB_START    = 35281  # Data from this index onwards serves as the knowledge base

# 3. Model & Training Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 3           
INPUT_DIM = 3       
BATCH_SIZE = 128    
EPOCHS = 50        
LR = 0.0001

# 4. Physical Constraints
# Force the predicted power to 0 MW when the actual wind speed is below this threshold
CUT_IN_SPEED_THRESHOLD = 2.5