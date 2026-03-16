import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import json
import os
import math

# --- SETTINGS & PATHS ---
BASE_DIR = "v3"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
DEVICE = "cpu"

st.set_page_config(page_title="GenRe_v3: Lending Loan Recourse", layout="wide")

# --- ARCHITECTURE DEFINITIONS ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class GenReV3(nn.Module):
    def __init__(self, cont_vocabs, cat_vocabs, emb_dim=64, layers=6, heads=4, ff_dim=256):
        super().__init__()
        self.num_cont, self.num_cat = len(cont_vocabs), len(cat_vocabs)
        self.seq_len = self.num_cont + self.num_cat
        self.sos_emb = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.cont_embs = nn.ModuleList([nn.Embedding(v, emb_dim) for v in cont_vocabs])
        self.cat_embs = nn.ModuleList([nn.Embedding(v, emb_dim) for v in cat_vocabs])
        self.pos_encoder = PositionalEncoding(emb_dim, max_len=self.seq_len + 1)
        
        # FIXED: Explicitly set dim_feedforward to match training checkpoint
        self.transformer = nn.Transformer(
            d_model=emb_dim, 
            nhead=heads, 
            num_encoder_layers=layers, 
            num_decoder_layers=layers, 
            dim_feedforward=ff_dim, 
            batch_first=True
        )
        self.cont_heads = nn.ModuleList([nn.Linear(emb_dim, v) for v in cont_vocabs])
        self.cat_heads = nn.ModuleList([nn.Linear(emb_dim, v) for v in cat_vocabs])

    def _get_embeddings(self, x_cont, x_cat):
        c_e = torch.stack([self.cont_embs[i](x_cont[:, i]) for i in range(self.num_cont)], dim=1)
        cat_e = torch.stack([self.cat_embs[i](x_cat[:, i]) for i in range(self.num_cat)], dim=1)
        return torch.cat([c_e, cat_e], dim=1)

    @torch.no_grad()
    def sample_recourse(self, src_cont_bins, src_cat, immutable_indices=None, increasing_indices=None, temp=0.5):
        self.eval()
        bs = src_cont_bins.size(0)
        memory = self.transformer.encoder(self.pos_encoder(self._get_embeddings(src_cont_bins, src_cat)))
        curr_embs = self.sos_emb.expand(bs, -1, -1)
        s_cont, s_cat = [], []
        
        # Default empty lists if none provided
        imm_idx = immutable_indices if immutable_indices else []
        inc_idx = increasing_indices if increasing_indices else []

        for i in range(self.seq_len):
            tgt = self.pos_encoder(curr_embs)
            mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)
            h = self.transformer.decoder(tgt, memory, tgt_mask=mask)[:, -1, :]
            
            # Determine if we are in continuous or categorical range
            is_cont = i < self.num_cont
            logits = self.cont_heads[i](h) if is_cont else self.cat_heads[i - self.num_cont](h)
            logits = logits / temp
            
            # --- APPLY CONSTRAINTS ---
            # Immutability: Force the original bin/value
            if i in imm_idx:
                val = src_cont_bins[:, i] if is_cont else src_cat[:, i - self.num_cont]
                choice = val.unsqueeze(-1)
            
            # Monotonicity (Increasing only): Mask bins lower than current
            elif i in inc_idx and is_cont:
                current_bin = src_cont_bins[:, i].item()
                # Create mask: -inf for all bins < current_bin
                m = torch.full_like(logits, float('-inf'))
                m[:, current_bin:] = 0 
                logits = logits + m
                choice = torch.argmax(logits, dim=-1, keepdim=True)
            
            else:
                choice = torch.argmax(logits, dim=-1, keepdim=True)

            # --- PREPARE NEXT STEP ---
            if is_cont:
                s_cont.append(choice)
                next_emb = self.cont_embs[i](choice.squeeze(-1)).unsqueeze(1)
            else:
                s_cat.append(choice)
                next_emb = self.cat_embs[i - self.num_cont](choice.squeeze(-1)).unsqueeze(1)
            
            curr_embs = torch.cat([curr_embs, next_emb], dim=1)
            
        return torch.cat(s_cont, dim=1), torch.cat(s_cat, dim=1)

class PaperANNProxy(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10), nn.ReLU(),
            nn.Linear(10, 10), nn.ReLU(),
            nn.Linear(10, 10), nn.ReLU(),
            nn.Linear(10, 1), nn.Sigmoid()
        )
    def forward(self, x_cont, x_cat):
        x = torch.cat([x_cont, x_cat.float()], dim=1)
        return self.net(x)

# --- ASSET LOADING ---

@st.cache_resource
def load_assets():
    with open(os.path.join(DATA_DIR, "meta.json"), 'r') as f:
        meta = json.load(f)
    
    bin_edges = np.load(os.path.join(DATA_DIR, "bin_edges_v3.npy"), allow_pickle=True)
    num_bins_per_feat = [len(e)-1 for e in bin_edges]
    
    num_scaler = joblib.load(os.path.join(DATA_DIR, "num_scaler.joblib"))
    X_cont_raw = np.load(os.path.join(DATA_DIR, "X_cont.npy"))
    X_cat_raw = np.load(os.path.join(DATA_DIR, "X_cat.npy"))
    y_raw = np.load(os.path.join(DATA_DIR, "y.npy"))

    # Load GenRe_v3 Checkpoint to extract architecture params
    genre_checkpoint = torch.load(os.path.join(MODEL_DIR, "genre_v3.pt"), map_location=DEVICE)
    
    # DYNAMIC CONFIGURATION: Extract params from checkpoint to avoid size mismatch
    emb_dim = genre_checkpoint.get('emb_dim', 64)
    layers = genre_checkpoint.get('layers', 6)
    heads = genre_checkpoint.get('heads', 4)
    ff_dim = genre_checkpoint.get('ff_dim', 256) # This was the cause of your error
    
    cat_vocab_list = list(meta['categorical_vocab_sizes'].values())
    
    genre = GenReV3(
        num_bins_per_feat, 
        cat_vocab_list, 
        emb_dim=emb_dim, 
        layers=layers, 
        heads=heads, 
        ff_dim=ff_dim
    )
    genre.load_state_dict(genre_checkpoint['model_state'])
    genre.eval()
    
    # Initialize ANN Proxy
    total_dim = meta['num_cont'] + meta['num_cat']
    ann = PaperANNProxy(total_dim)
    ann_checkpoint = torch.load(os.path.join(MODEL_DIR, "ann_flexible.pt"), map_location=DEVICE)
    ann.load_state_dict(ann_checkpoint['model_state'])
    ann.eval()
    
    return meta, bin_edges, num_scaler, genre, ann, X_cont_raw, X_cat_raw, y_raw

# --- EXECUTION ---

meta, bin_edges, scaler, genre, ann, X_cont_raw, X_cat_raw, y_raw = load_assets()

# --- SESSION STATE & UI ---
if 'fact_cont' not in st.session_state:
    neg_indices = np.where(y_raw == 0)[0]
    idx = np.random.choice(neg_indices)
    st.session_state.fact_cont = X_cont_raw[idx].tolist()
    st.session_state.fact_cat = X_cat_raw[idx].tolist()

def load_random_user():
    neg_indices = np.where(y_raw == 0)[0]
    idx = np.random.choice(neg_indices)
    st.session_state.fact_cont = X_cont_raw[idx].tolist()
    st.session_state.fact_cat = X_cat_raw[idx].tolist()

st.title("🏦 GenRe_v3: Lending Loan Recourse")
st.sidebar.button("🎲 Load Random Rejected User", on_click=load_random_user)

tab1, tab2, tab3 = st.sidebar.tabs(["Loan & Identity", "Financials", "Credit Profile"])

with tab1:
    st.subheader("Loan Information")
    for i, name in enumerate(meta['categorical_features']):
        v_map = meta['categorical_value_maps'][name]
        options = list(v_map.keys())
        current_code = int(st.session_state.fact_cat[i])
        current_str = [k for k, v in v_map.items() if v == current_code][0]
        choice = st.selectbox(f"{name}", options, index=options.index(current_str), disabled=(name in meta['immutable_features']))
        st.session_state.fact_cat[i] = v_map[choice]

with tab2:
    st.subheader("Income & Debt")
    for i in range(20):
        name = meta['continuous_features'][i]
        st.session_state.fact_cont[i] = st.slider(f"{name}", 0.0, 1.0, float(st.session_state.fact_cont[i]), step=0.01)

with tab3:
    st.subheader("History & Utilization")
    for i in range(20, 67):
        name = meta['continuous_features'][i]
        st.session_state.fact_cont[i] = st.slider(f"{name}", 0.0, 1.0, float(st.session_state.fact_cont[i]), step=0.01)

# --- INFERENCE ---
x_fact_cont = torch.tensor([st.session_state.fact_cont], dtype=torch.float32)
x_fact_cat = torch.tensor([st.session_state.fact_cat], dtype=torch.long)

col_status, col_recourse = st.columns([1, 2])

with col_status:
    st.header("Decision Status")
    prob = ann(x_fact_cont, x_fact_cat).item()
    status = "✅ APPROVED" if prob > 0.5 else "❌ REJECTED"
    st.markdown(f"<h1 style='text-align: center; color: {'#2ecc71' if prob > 0.5 else '#e74c3c'};'>{status}</h1>", unsafe_allow_html=True)
    st.progress(prob)
    st.write(f"Approval Probability: **{prob*100:.2f}%**")

with col_recourse:
    st.header("Actionable Recourse Path")
    if prob > 0.5:
        st.balloons()
        st.success("User is already approved. No action needed.")
    else:
        # 1. PREPARE INPUTS: Binner (Continuous values -> Bin indices)
        x_fact_bins = []
        for i in range(67):
            idx = np.digitize(st.session_state.fact_cont[i], bin_edges[i]) - 1
            idx = np.clip(idx, 0, len(bin_edges[i]) - 2)
            x_fact_bins.append(idx)
        
        src_bins_t = torch.tensor([x_fact_bins], dtype=torch.long)

        # 2. IDENTIFY CONSTRAINT INDICES
        # We find which indices in the sequence (cont + cat) are restricted
        imm_indices = []
        inc_indices = []
        
        # Check continuous features (Indices 0 to 66)
        for i, name in enumerate(meta['continuous_features']):
            if name in meta.get('immutable_features', []):
                imm_indices.append(i)
            # Optional: handle features that can only increase (e.g., age, tenure)
            if name in meta.get('increasing_features', []):
                inc_indices.append(i)
        
        # Check categorical features (Indices 67 onwards)
        for i, name in enumerate(meta['categorical_features']):
            seq_idx = i + 67 
            if name in meta.get('immutable_features', []):
                imm_indices.append(seq_idx)

        # GENERATE CONSTRAINED RECOURSE
        with st.spinner("Calculating optimal constrained path..."):
            rec_bins, rec_cat = genre.sample_recourse(
                src_bins_t, 
                x_fact_cat, 
                immutable_indices=imm_indices,
                increasing_indices=inc_indices,
                temp=0.5
            )
        
        # DECODE BINS BACK TO SCALED VALUES
        rec_cont_scaled = []
        for i in range(67):
            edges = bin_edges[i]
            b_idx = rec_bins[0, i].item()
            midpoint = (edges[b_idx] + edges[b_idx+1]) / 2.0
            rec_cont_scaled.append(midpoint)
        
        rec_cont_t = torch.tensor([rec_cont_scaled], dtype=torch.float32)
        new_prob = ann(rec_cont_t, rec_cat).item()
        
        st.subheader("Recommended Steps")
        changes = []
        for i, name in enumerate(meta['continuous_features']):
            diff = rec_cont_scaled[i] - st.session_state.fact_cont[i]
            if abs(diff) > 0.05:
                changes.append({"Feature": name, "Action": "Increase" if diff > 0 else "Decrease", "Intensity": f"{abs(diff)*100:.1f}% change"})
        
        for i, name in enumerate(meta['categorical_features']):
            if rec_cat[0, i].item() != st.session_state.fact_cat[i]:
                v_map = meta['categorical_value_maps'][name]
                new_val_str = [k for k, v in v_map.items() if v == rec_cat[0, i].item()][0]
                changes.append({"Feature": name, "Action": "Switch to", "Intensity": new_val_str})

        if changes:
            st.table(pd.DataFrame(changes))
            st.metric("Approval Probability", f"{new_prob*100:.2f}%", delta=f"{(new_prob-prob)*100:.2f}%")
        else:
            st.warning("Could not find a realistic recourse path for this specific profile. Try adjusting inputs.")

st.divider()
st.caption("GenRe_v3 | Powered by Dynamic Regression Tree Binning")