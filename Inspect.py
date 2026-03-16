import os
import json
import torch
import numpy as np
import joblib
from pprint import pprint

# Define the base directory (assuming you've downloaded the 'v3' folder)
BASE_DIR = "v3"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

def inspect_deployment_assets():
    print("="*60)
    print(" GENRE_V3 DEPLOYMENT ASSET INSPECTION")
    print("="*60)

    # 1. Inspect meta.json (The Heart of the Architecture)
    meta_path = os.path.join(DATA_DIR, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print("\n[1] METADATA (meta.json):")
        pprint(meta)
    else:
        print("\n[!] ERROR: meta.json not found.")

    # 2. Inspect bin_edges_v3.npy (The Dynamic Ruler)
    edges_path = os.path.join(DATA_DIR, "bin_edges_v3.npy")
    if os.path.exists(edges_path):
        # allow_pickle=True is needed if saved as a list of arrays
        bin_edges = np.load(edges_path, allow_pickle=True)
        print(f"\n[2] BIN EDGES (bin_edges_v3.npy):")
        print(f"    - Number of continuous features binned: {len(bin_edges)}")
        for i, edge in enumerate(bin_edges):
            print(f"      Feature {i} bin count: {len(edge)-1}")
    
    # 3. Inspect Preprocessing Artifacts
    print("\n[3] PREPROCESSING (Joblibs):")
    for job in ["num_scaler.joblib", "cat_imputer.joblib", "num_imputer.joblib"]:
        path = os.path.join(DATA_DIR, job)
        if os.path.exists(path):
            obj = joblib.load(path)
            print(f"    - {job}: Loaded successfully ({type(obj).__name__})")
            if hasattr(obj, 'feature_names_in_'):
                print(f"      Features: {list(obj.feature_names_in_)}")

    # 4. Inspect Model Weights (PyTorch)
    print("\n[4] MODEL WEIGHTS (.pt):")
    for model_file in ["genre_v3.pt", "ann_flexible.pt"]:
        path = os.path.join(MODEL_DIR, model_file)
        if os.path.exists(path):
            # Load on CPU for inspection
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            print(f"    - {model_file}: Loaded.")
            # Check keys to verify if it's a full model or just state_dict
            if isinstance(state_dict, dict):
                print(f"      Architecture detected: state_dict with {len(state_dict.keys())} layers")
                # Peek at embedding layer to confirm vocab size
                first_key = list(state_dict.keys())[0]
                print(f"      Sample Layer [{first_key}]: Shape {len(state_dict[first_key])}")

    # 5. Data Shapes (Optional but good for sanity)
    print("\n[5] DATA ARRAYS (.npy):")
    for npy in ["X_cont.npy", "X_cat.npy", "y.npy"]:
        path = os.path.join(DATA_DIR, npy)
        if os.path.exists(path):
            data = np.load(path)
            print(f"    - {npy}: Shape {data.shape}")

if __name__ == "__main__":
    inspect_deployment_assets()