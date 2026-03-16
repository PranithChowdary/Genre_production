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

# --- CORE ARCHITECTURES ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class GenReV3(nn.Module):
    def __init__(self, cont_vocabs, cat_vocabs, emb_dim=64, layers=6, heads=4, ff_dim=256):
        super().__init__()
        self.num_cont, self.num_cat = len(cont_vocabs), len(cat_vocabs)
        self.seq_len = self.num_cont + self.num_cat
        self.sos_emb = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.cont_embs = nn.ModuleList([nn.Embedding(v, emb_dim) for v in cont_vocabs])
        self.cat_embs = nn.ModuleList([nn.Embedding(v, emb_dim) for v in cat_vocabs])
        self.pos_encoder = PositionalEncoding(emb_dim, self.seq_len + 1)
        self.transformer = nn.Transformer(d_model=emb_dim, nhead=heads, num_encoder_layers=layers, 
                                          num_decoder_layers=layers, dim_feedforward=ff_dim, batch_first=True)
        self.cont_heads = nn.ModuleList([nn.Linear(emb_dim, v) for v in cont_vocabs])
        self.cat_heads = nn.ModuleList([nn.Linear(emb_dim, v) for v in cat_vocabs])

    def _embed(self, x_cont, x_cat):
        c_e = torch.stack([self.cont_embs[i](x_cont[:, i]) for i in range(self.num_cont)], dim=1)
        cat_e = torch.stack([self.cat_embs[i](x_cat[:, i]) for i in range(self.num_cat)], dim=1)
        return torch.cat([c_e, cat_e], dim=1)

    @torch.no_grad()
    def sample_algorithm2(self, src_bins, src_cat, temp=0.1):
        bs = src_bins.size(0)
        memory = self.transformer.encoder(self.pos_encoder(self._embed(src_bins, src_cat)))
        curr = self.sos_emb.expand(bs, -1, -1)
        s_cont, s_cat = [], []
        for i in range(self.seq_len):
            tgt = self.pos_encoder(curr)
            mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            h = self.transformer.decoder(tgt, memory, tgt_mask=mask)[:, -1, :]
            if i < self.num_cont:
                logits = self.cont_heads[i](h) / temp
                choice = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                s_cont.append(choice)
                next_emb = self.cont_embs[i](choice.squeeze(-1)).unsqueeze(1)
            else:
                idx = i - self.num_cont
                logits = self.cat_heads[idx](h) / temp
                choice = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                s_cat.append(choice)
                next_emb = self.cat_embs[idx](choice.squeeze(-1)).unsqueeze(1)
            curr = torch.cat([curr, next_emb], dim=1)
        return torch.cat(s_cont, dim=1), torch.cat(s_cat, dim=1)

class FlexibleANNProxy(nn.Module):
    def __init__(self, num_cont, cat_vocabs, emb_dim=32, h1=256, h2=128):
        super().__init__()
        self.cont_proj = nn.Linear(num_cont, emb_dim)
        self.cat_embeddings = nn.ModuleList([nn.Embedding(v, emb_dim) for v in cat_vocabs])
        input_dim = (1 + len(cat_vocabs)) * emb_dim
        self.classifier = nn.Sequential(nn.Linear(input_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1), nn.Sigmoid())

    def forward(self, x_cont, x_cat):
        c = self.cont_proj(x_cont).unsqueeze(1)
        cat = torch.stack([self.cat_embeddings[i](x_cat[:, i]) for i in range(len(self.cat_embeddings))], dim=1)
        combined = torch.cat([c, cat], dim=1).view(x_cont.size(0), -1)
        return self.classifier(combined)

# --- UTILITIES ---

def safe_inverse_transform(binner, X_bins, meta):
    results = []
    for i, b in enumerate(X_bins):
        edges = binner[i]
        max_valid_bin = len(edges) - 2
        safe_b = int(np.clip(b, 0, max_valid_bin))
        val = (edges[safe_b] + edges[safe_b+1]) / 2.0
        name = meta['continuous_features'][i]
        if any(k in name.lower() for k in ["acc", "inq", "pub", "delinq", "num", "mort", "collections"]):
            val = round(val)
        results.append(val)
    return results

def format_financial(name, val):
    unit = "$" if any(k in name.lower() for k in ["amnt", "inc", "bal", "lim", "pymnt", "installment"]) else ""
    pct = "%" if any(k in name.lower() for k in ["rate", "util", "pct"]) else ""
    return f"{unit}{val:,.2f}{pct}"

@st.cache_resource
def load_assets():
    with open(os.path.join(DATA_DIR, "meta.json")) as f: meta = json.load(f)
    bin_edges = np.load(os.path.join(DATA_DIR, "bin_edges_v3.npy"), allow_pickle=True)
    scaler = joblib.load(os.path.join(DATA_DIR, "num_scaler.joblib"))
    X_cont = np.load(os.path.join(DATA_DIR, "X_cont.npy"))
    X_cat = np.load(os.path.join(DATA_DIR, "X_cat.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    
    checkpoint = torch.load(os.path.join(MODEL_DIR, "genre_v3.pt"), map_location="cpu")
    emb_dim = checkpoint.get('emb_dim', 64)
    layers = checkpoint.get('layers', 6)
    heads = checkpoint.get('heads', 4)
    ff_dim = checkpoint.get('ff_dim', 256)
    
    cat_vocabs = [meta['categorical_vocab_sizes'][f] for f in meta['categorical_features']]
    genre = GenReV3([len(e)-1 for e in bin_edges], cat_vocabs, emb_dim=emb_dim, layers=layers, heads=heads, ff_dim=ff_dim)
    genre.load_state_dict(checkpoint["model_state"])
    genre.eval()
    
    ann = FlexibleANNProxy(meta['num_cont'], cat_vocabs, emb_dim=32)
    ann.load_state_dict(torch.load(os.path.join(MODEL_DIR, "ann_flexible.pt"), map_location="cpu")["model_state"])
    ann.eval()
    return meta, bin_edges, scaler, genre, ann, X_cont, X_cat, y

meta, bin_edges, scaler, genre, ann, X_cont, X_cat, y_all = load_assets()

# --- CONSTRAINED RECOURSE GENERATION ---
@torch.no_grad()
def generate_demo_recourse(fact_bins, fact_cat, k=100, lam=0.01, search_radius=3):
    best_bins, best_cat = fact_bins.clone(), fact_cat.clone()
    f_norm = torch.tensor([safe_inverse_transform(bin_edges, fact_bins[0].tolist(), meta)], dtype=torch.float32)
    p_init = ann(f_norm, fact_cat).item()
    best_prob, best_score = p_init, -1e9

    # Directional Constraints
    increasing_only = ["annual_inc", "total_rev_hi_lim", "avg_cur_bal", "tot_cur_bal"]
    decreasing_only = ["dti", "revol_util", "loan_amnt", "inq_last_6mths", "pub_rec_bankruptcies"]

    for i_sample in range(k):
        temp = 0.1 + (i_sample / k) * 0.6
        gen_bins, gen_cats = genre.sample_algorithm2(fact_bins, fact_cat, temp=temp)
        
        diff = gen_bins - fact_bins
        
        # Apply Constraints to Continuous
        for i in range(meta['num_cont']):
            name = meta['continuous_features'][i]
            if name in meta['immutable_features']:
                gen_bins[:, i] = fact_bins[:, i]
            else:
                # Use the dynamic search_radius passed from UI
                clamped_val = fact_bins[:, i] + torch.clamp(diff[:, i], -search_radius, search_radius)
                
                if name in increasing_only:
                    gen_bins[:, i] = torch.max(clamped_val, fact_bins[:, i])
                elif name in decreasing_only:
                    gen_bins[:, i] = torch.min(clamped_val, fact_bins[:, i])
                else:
                    gen_bins[:, i] = clamped_val
                
                gen_bins[:, i] = torch.clamp(gen_bins[:, i], 0, len(bin_edges[i]) - 2)

        # Apply Constraints to Categorical
        for i, name in enumerate(meta['categorical_features']):
            if name in meta['immutable_features']:
                gen_cats[:, i] = fact_cat[:, i]

        gn_norm_vals = safe_inverse_transform(bin_edges, gen_bins[0].tolist(), meta)
        gn_norm = torch.tensor([gn_norm_vals], dtype=torch.float32)
        probs = ann(gn_norm, gen_cats).item()
        
        n_cost = torch.abs(gen_bins - fact_bins).float().sum()
        c_cost = (gen_cats != fact_cat).float().sum() * 5.0
        total_cost = (n_cost + c_cost) * 0.05

        if probs < 0.90:
            score = probs * 100.0 - (lam * total_cost)
        else:
            score = 1000.0 + (probs * 10.0) - total_cost

        if score > best_score:
            best_bins, best_cat, best_score, best_prob = gen_bins, gen_cats, score, probs
            
    return best_bins, best_cat, best_prob

# --- UI ---

st.set_page_config(page_title="SBI Smart Recourse", layout="wide")
st.title("🏦 SBI Smart Recourse")

if 'view_mode' not in st.session_state: st.session_state.view_mode = "Individual"
if 'user_idx' not in st.session_state: st.session_state.user_idx = None

with st.sidebar:
    st.header("Navigation")
    col_nav1, col_nav2 = st.columns(2)
    if col_nav1.button("👤 Individual"): st.session_state.view_mode = "Individual"
    if col_nav2.button("📊 Bulk Mode"): st.session_state.view_mode = "Bulk"
    
    st.divider()
    
    if st.session_state.view_mode == "Individual":
        st.header("Applicant Selection")
        if st.button("🎲 Load Random Rejected User"):
            neg_indices = np.where(y_all == 0)[0]
            subset = np.random.choice(neg_indices, 1000)
            sub_xc, sub_xcat = torch.tensor(X_cont[subset], dtype=torch.float32), torch.tensor(X_cat[subset], dtype=torch.long)
            with torch.no_grad():
                probs = ann(sub_xc, sub_xcat).squeeze().numpy()
            rejected = subset[probs < 0.35]
            if len(rejected) > 0:
                st.session_state.user_idx = int(np.random.choice(rejected))

if st.session_state.view_mode == "Individual" and st.session_state.user_idx is not None:
    idx = st.session_state.user_idx
    f_cont_scaled, f_cat = X_cont[idx:idx+1], X_cat[idx:idx+1]
    f_real = scaler.inverse_transform(f_cont_scaled)[0]
    
    with torch.no_grad():
        p_fact = ann(torch.tensor(f_cont_scaled, dtype=torch.float32), torch.tensor(f_cat, dtype=torch.long)).item()
    
    st.write(f"### Applicant ID: {idx} - Current Approval Probability: **{p_fact:.2%}**")

    # Advisory Configuration (Timeframe & Effort)
    with st.expander("🛠️ STEP 2: Advisory Configuration", expanded=True):
        st.info("Define how much time and effort the applicant can invest.")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            timeframe = st.selectbox(
                "Target Actionable Timeframe", 
                options=[3, 6, 12], 
                format_func=lambda x: f"{x} Months", 
                index=1,
                key="ui_timeframe"
            )
        with col_c2:
            effort = st.slider(
                "Allowable Effort (Intensity)", 
                min_value=1, max_value=10, value=5, 
                help="Higher intensity allows larger financial changes."
            )
        
        # Calculate dynamic search radius for model based on inputs
        # 3mo = 2-3 bins, 12mo = 5-6 bins
        dynamic_radius = int((timeframe / 3.0) + (effort / 3.0))
        
        st.info(f"💡 Strategy: Seeking improvement with a **Search Radius of {dynamic_radius} bins** over **{timeframe} months**.")

        tab_num, tab_cat = st.tabs(["Allowed Financial Ranges", "Allowed Transitions"])
        
        with tab_num:
            priority = ["loan_amnt", "annual_inc", "dti", "installment", "revol_util"]
            cols = st.columns(2)
            for i, name in enumerate(priority):
                if name in meta["continuous_features"] and name not in meta["immutable_features"]:
                    f_idx = meta["continuous_features"].index(name)
                    curr_val = float(f_real[f_idx])
                    with cols[i % 2]:
                        st.slider(f"{name} (Current: {format_financial(name, curr_val)})", 0.0, 1.0, (0.0, 1.0), key=f"range_{name}")

        with tab_cat:
            for i, name in enumerate(meta["categorical_features"]):
                if name not in meta["immutable_features"]:
                    v_map = meta["categorical_value_maps"][name]
                    options = list(v_map.keys())
                    curr_val_str = [k for k,v in v_map.items() if v == f_cat[0, i]][0]
                    st.multiselect(f"Allowed values for {name}", options, default=[curr_val_str], key=f"cat_{name}")

    if st.button("🚀 Generate Recourse"):
        f_bins = [np.clip(np.digitize(f_cont_scaled[0, i], bin_edges[i]) - 1, 0, len(bin_edges[i])-2) for i in range(len(f_real))]
        f_bins_t = torch.tensor([f_bins], dtype=torch.long)
        
        with st.spinner("Finding optimal recourse path..."):
            r_bins, r_cat, p_end = generate_demo_recourse(
                f_bins_t, 
                torch.tensor(f_cat, dtype=torch.long), 
                k=150, 
                search_radius=dynamic_radius
            )
            
            # Post-check Categorical
            for i, name in enumerate(meta["categorical_features"]):
                if name not in meta["immutable_features"]:
                    allowed = st.session_state.get(f"cat_{name}", [])
                    v_map = meta["categorical_value_maps"][name]
                    rec_str = [k for k, v in v_map.items() if v == r_cat[0, i].item()][0]
                    if allowed and rec_str not in allowed:
                        r_cat[0, i] = f_cat[0, i]

        st.subheader("✅ Actionable Recommendations")
        m1, m2 = st.columns(2)
        m1.metric("Final Prob", f"{p_end:.2%}", delta=f"{(p_end-p_fact):.2%}")
        m2.metric("Result", "APPROVED" if p_end > 0.5 else "IMPROVED")

        r_cont_vals = safe_inverse_transform(bin_edges, r_bins[0].tolist(), meta)
        r_real = scaler.inverse_transform([r_cont_vals])[0]
        
        changes = []
        for i, name in enumerate(meta["continuous_features"]):
            if abs(r_real[i] - f_real[i]) > 1e-2:
                changes.append({"Feature": name, "Current": format_financial(name, f_real[i]), "Suggested": format_financial(name, r_real[i]), "Change": "Increase" if r_real[i] > f_real[i] else "Decrease"})
        
        for i, name in enumerate(meta["categorical_features"]):
            if r_cat[0, i] != f_cat[0, i]:
                inv = {v: k for k, v in meta["categorical_value_maps"][name].items()}
                changes.append({"Feature": name, "Current": inv[int(f_cat[0, i])], "Suggested": inv[int(r_cat[0, i])], "Change": "Switch"})

        if changes:
            st.dataframe(pd.DataFrame(changes), use_container_width=True)
        else:
            st.warning("No valid recourse path found within the defined timeframe/effort constraints.")

elif st.session_state.view_mode == "Bulk":
    st.header("📊 Bulk Recourse Analysis Report")
    if st.session_state.bulk_results is not None:
        df = st.session_state.bulk_results
        u_data = st.session_state.bulk_recourses
        
        # Summary Metrics
        avg_prob = df['Score'].mean()
        success_rate = (df['Score'] > 0.5).mean()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Processed", len(df))
        m2.metric("Approval Success Rate", f"{success_rate:.0%}")
        m3.metric("Avg. Target Probability", f"{avg_prob:.2%}")
        
        st.divider()
        st.subheader("Batch Results Detail")
        st.dataframe(df.drop(columns=['Score']), use_container_width=True)
        st.dataframe(u_data, width="stretch")
        
        # Download
        csv = u_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Report (CSV)", csv, "bulk_recourse_report.csv", "text/csv")
    else:
        st.info("Configure batch settings in the sidebar and click 'Generate Bulk Recourse' to begin.")
else:
    st.info("Pick a user from the sidebar to begin.")