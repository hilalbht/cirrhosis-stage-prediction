import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================
# SAYFA AYARLARI
# =========================
st.set_page_config(
    page_title="Klinik Parametrelere Dayalı Siroz Evre Tahmin Sistemi",
    layout="centered"
)

# =========================
# STİL (CSS)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

:root {
    --primary-red: #8b1e1e;
    --primary-red-dark: #661414;
}

/* ===== GENEL ===== */
.stApp {
    background: linear-gradient(90deg,
        rgba(33,33,110,1) 0%,
        rgba(31,42,118,1) 40%,
        rgba(15,60,120,1) 100%);
    color: #f2f4f8;
    font-family: 'Inter', sans-serif;
}

/* ===== BAŞLIK KARTI ===== */
.header-card {
    background: rgba(15, 42, 68, 0.55);
    backdrop-filter: blur(6px);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 20px;
}

.main-title {
    font-family: "Times New Roman", Georgia, serif;
    text-align: center;
    color: #f2f4f8;
}

/* ===== SLIDER (KALIN + TEK RENK) ===== */
div[data-baseweb="slider"] > div {
    padding: 8px 0;
}

div[data-baseweb="slider"] > div > div {
    height: 14px !important;
    background-color: var(--primary-red) !important;
}

div[data-baseweb="slider"] span {
    width: 26px !important;
    height: 26px !important;
    background-color: var(--primary-red) !important;
    border: none !important;
}

/* Pasif track */
div[data-baseweb="slider"] div[aria-hidden="true"] {
    background-color: var(--primary-red) !important;
}

/* ===== RADIO BUTTON ===== */
div[role="radiogroup"] label {
    transform: scale(1.1);
    margin-right: 12px;
}

div[role="radiogroup"] input:checked + div {
    background-color: var(--primary-red) !important;
    border-color: var(--primary-red) !important;
}

/* ===== SELECTBOX ===== */
div[data-baseweb="select"] > div {
    border-color: var(--primary-red) !important;
}

/* ===== PROGRESS BAR ===== */
.stProgress > div > div > div {
    background-color: var(--primary-red) !important;
}

/* ===== BUTON ===== */
.stButton > button {
    background-color: var(--primary-red);
    color: #ffffff;
    border-radius: 18px;
    padding: 16px 70px;
    font-size: 22px;
    font-weight: 800;
    transition: transform 0.25s ease;
}

.stButton > button:hover {
    background-color: var(--primary-red-dark);
    transform: scale(1.07);
}

/* ===== SONUÇ KARTI ===== */
.result-card {
    background: linear-gradient(135deg, #0f2a44, #123a5f);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.25);
}

/* ===== TABLO ===== */
.custom-table {
    background-color: rgba(15, 42, 68, 0.85);
    border-radius: 16px;
    padding: 20px;
    margin-top: 25px;
}

.custom-table th {
    color: #ffb3b3;
}
</style>
""", unsafe_allow_html=True)

# =========================
# MODEL
# =========================
model = joblib.load("xgboost_stage_model.pkl")
le_stage = joblib.load("stage_label_encoder.pkl")

# =========================
# BAŞLIK
# =========================
st.markdown("""
<div class="header-card">
    <h1 class="main-title">
    Klinik Parametrelere Dayalı<br>Siroz Evre Tahmin Sistemi
    </h1>
    <p style="text-align:center;">
    Eğitim ve klinik simülasyon amaçlı geliştirilmiştir.
    </p>
    <p style="text-align:center; font-size:14px;">
    ⚠️ Bu sistem tanı koymaz, klinik kararların yerine geçmez.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# GİRDİLER
# =========================
st.subheader("Demografik Bilgiler")
age = st.slider("Yaş", 1, 100, 50)
sex = st.radio("Cinsiyet", ["Female", "Male"], horizontal=True)

st.divider()

st.subheader("Takip ve Tedavi Bilgileri")
n_days = st.slider("Takip Süresi (N_Days)", 0, 5000, 1000)
status = st.radio("Hasta Durumu (Status)", ["C", "CL", "D"], horizontal=True)
drug = st.radio("Uygulanan Tedavi (Drug)", ["Placebo", "D-penicillamine"], horizontal=True)

st.divider()

st.subheader("Klinik Bulgular")
ascites = st.selectbox("Ascites", ["Yok", "Var"])
hepatomegaly = st.selectbox("Hepatomegaly", ["Yok", "Var"])
spiders = st.selectbox("Spiders", ["Yok", "Var"])
edema = st.selectbox("Edema", ["0", "1", "2"])

st.divider()

st.subheader("Laboratuvar Bulguları")
bilirubin = st.slider("Bilirubin", 0.1, 30.0, 1.0)
cholesterol = st.slider("Cholesterol", 100.0, 500.0, 250.0)
albumin = st.slider("Albumin", 1.0, 6.0, 3.5)
copper = st.slider("Copper", 0.0, 300.0, 50.0)
alk_phos = st.slider("Alk_Phos", 50.0, 3000.0, 500.0)
sgot = st.slider("SGOT", 10.0, 500.0, 50.0)
trig = st.slider("Tryglicerides", 50.0, 500.0, 150.0)
platelets = st.slider("Platelets", 50.0, 500.0, 250.0)
prothrombin = st.slider("Prothrombin", 8.0, 20.0, 12.0)

st.divider()

# =========================
# BUTON
# =========================
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_btn = st.button("EVRE TAHMİNİ YAP")

# =========================
# TAHMİN
# =========================
if predict_btn:
    sex_val = 1 if sex == "Male" else 0
    status_val = {"C":0, "CL":1, "D":2}[status]
    drug_val = 1 if drug == "D-penicillamine" else 0

    input_df = pd.DataFrame([{
        "N_Days": n_days,
        "Status": status_val,
        "Drug": drug_val,
        "Age": age,
        "Sex": sex_val,
        "Ascites": 1 if ascites == "Var" else 0,
        "Hepatomegaly": 1 if hepatomegaly == "Var" else 0,
        "Spiders": 1 if spiders == "Var" else 0,
        "Edema": int(edema),
        "Bilirubin": bilirubin,
        "Cholesterol": cholesterol,
        "Albumin": albumin,
        "Copper": copper,
        "Alk_Phos": alk_phos,
        "SGOT": sgot,
        "Tryglicerides": trig,
        "Platelets": platelets,
        "Prothrombin": prothrombin,
        "Status_label": status_val,
        "Drug_label": drug_val
    }])[model.feature_names_in_]

    pred = model.predict(input_df)
    probs = model.predict_proba(input_df)[0]
    stage = le_stage.inverse_transform(pred)[0]

    st.markdown(f"""
    <div class="result-card">
        <h2>Tahmin Edilen Siroz Evresi</h2>
        <h1 style="font-size:46px;">Stage {stage}</h1>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Evre Olasılıkları")
    for s, p in zip(le_stage.classes_, probs):
        st.progress(float(p), text=f"Stage {s}: %{p*100:.2f}")
