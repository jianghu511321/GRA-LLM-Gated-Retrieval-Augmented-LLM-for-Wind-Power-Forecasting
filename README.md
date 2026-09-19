# GRA-LLM-Gated-Retrieval-Augmented-LLM-for-Wind-Power-Forecasting
This project is the implementation of the paper "GRA-LLM: A Gated Retrieval-Augmented Large Language Model Framework for Ultra-Short-Term Wind Power Forecasting".

# Project Introduction
Ultra-short-term wind power forecasting (WPF) plays a vital role in real-time power grid operation, dynamic active power dispatch, and frequency regulation. However, existing deep learning and Large Language Model (LLM)-based paradigms suffer from three fundamental bottlenecks when deployed in real-world wind farm scenarios:
1. **Historical Reference Deficit ("Closed-book Trap"):** Purely parametric models rely solely on internal weights, lacking explicit external references during sudden wind ramps.
2. **Retrieval Negative Transfer ("Pseudo-similar Vulnerability"):** High-frequency meteorological noise leads distance-based time-series retrieval to recall pseudo-similar sequences, triggering catastrophic performance degradation.
3. **Physical Constraints Absence:** Purely data-driven models lack domain boundaries, generating non-physical micro-oscillations during low-wind conditions instead of absolute zero outputs.

To bridge these gaps, **GRA-LLM** integrates a continuous FAISS-based time-series retrieval module, a fine-grained dynamic trust gate, and physical cut-in wind speed constraints into an autoregressive LLM backbone.

## System Architecture
The overall framework operates as follow:

<img width="3988" height="1894" alt="架构图" src="https://github.com/user-attachments/assets/03187d0d-e779-479e-b276-83360153eea9" />

1. **FAISS-based Time-Series Retriever:** Converts lookback window sequences (`seq_len=96`) into 1D vectors and uses `faiss.IndexFlatL2` to perform exact high-speed searching for Top-K analogous historical scenarios.
2. **Attention-based Feature Fusion:** Utilizes Scaled Dot-Product Attention to calculate interaction scores between the current query and retrieved historical representations, adaptively aggregating external context.
3. **Fine-Grained Dynamic Trust Gate:** A Multi-Layer Perceptron (MLP) with Sigmoid activation that computes a scalar gate value `g` in `[0, 1]`, evaluating the correlation between the query and historical references to control knowledge injection.
4. **Autoregressive LLM Backbone:** Built on a lightweight GPT-2 architecture (`d_model=128`, 4 layers, 4 heads) with causal masks to capture temporal dependencies.
5. **Physical Boundary Constraints:** A post-processing rule that monitors actual wind speed at the forecasting step and strictly forces power output to zero when wind speed falls below the 2.5 m/s cut-in threshold.

## Hardware Used
The model training and inference were conducted on the following environment setup:
* **GPU:** NVIDIA GeForce RTX 5090 (Support for CUDA acceleration)
* **CPU:** Intel Core i9-13900K @ 3.00 GHz
* **Memory (RAM):** 64 GB DDR5

---

## Software Stack
* **Language:** Python 3.8+
* **Deep Learning Framework:** PyTorch (`torch`, `torch.nn.functional`)
* **Language Model Backbone:** HuggingFace `transformers` (`GPT2Model`, `GPT2Config`)
* **Vector Indexing & Retrieval:** `faiss-cpu` / `faiss-gpu`
* **Data Processing & Utilities:** `pandas`, `numpy`, `scikit-learn` (`MinMaxScaler`), `joblib`, `matplotlib`

## Dataset
This project uses the SDWPF dataset from the [Baidu KDD Cup 2022 (AI Studio Competition 152)](https://aistudio.baidu.com/competition/detail/152/0/introduction),
containing SCADA records from 134 wind turbines (Longyuan Power Group), 
sampled every 10 minutes.

Relevant columns (0-indexed) used by this project:
| Index | Column | Meaning |
|---|---|---|
| 3 | Wspd | Wind Speed |
| 4 | Wdir | Wind Direction |
| 12 | Patv | Active Power |

Download the dataset from the competition page and place it as `./data/wind_power_data.csv` before running the pipeline.

 ##  How to Run
Follow these steps to set up and run the GRA-LLM framework on your local machine.

**Step 1: Clone the repository and install dependencies**
```
git clone [https://github.com/jianghu511321/GRA-LLM-Gated-Retrieval-Augmented-LLM-for-Wind-Power-Forecasting
.git]
cd GRA-LLM-Gated-Retrieval-Augmented-LLM-for-Wind-Power-Forecasting
pip install -r requirements.txt
```

**Step 2: Prepare the dataset**
Create a data folder in the root directory and place your wind power dataset inside it:
```
your-repo-name/
 ├── data/
 │    └── wind_power_data.csv   <-- Place your CSV here
 ├── config.py
 ├── ...
```
(Note: Ensure your CSV has at least the columns for Wind Speed, Wind Direction, and Active Power. Modify COL_INDICES in config.py if your column order differs).

**Step 3: Run Data Preprocessing**
This step will normalize the data, build the knowledge base, and generate sequence files.
```
python data_preprocessing.py
```

**Step 4: Train and Evaluate the Model**
Run the main pipeline. This will train the GPT-2 based model, perform FAISS retrieval, apply physical constraints, and output the final predictions.
```
python main.py
```



## Test Result
The model is evaluated using RMSE (Root Mean Square Error) and MAE (Mean Absolute Error).
By integrating the Retrieval-Augmented Framework (RAF) with the Dynamic Trust Gate and applying domain-specific Physical Constraints (Cut-in wind speed rule), the model achieves highly accurate predictions:
| Model Configuration | MAE (kW) | RMSE (kW) |
| :--- | :---: | ---: |
| RNN | 44.23 | 76.28 |
| LSTM | 39.79 | 73.25 |
| PatchTST | 40.49 | 76.56 |
| TimeLLM | 45.49 | 83.05 |
| **GRA-LLM** | **32.77** | **64.02** |
---
