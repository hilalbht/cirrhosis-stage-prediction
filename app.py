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

.stApp {
    background: linear-gradient(90deg,
        rgba(2,0,36,1) 0%,
        rgba(9,9,121,1) 51%,
        rgba(0,212,255,1) 100%);
    color: #f2f4f8;
    font-family: 'Inter', sans-serif;
}

/* ===== BAŞLIK KARTI ===== */
.header-card {
    background: rgba(10, 25, 60, 0.65);
    backdrop-filter: blur(6px);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 25px;
    transition: transform 0.3s ease;
}

.header-card:hover {
    transform: scale(1.03);
}

.main-title {
    font-family: "Times New Roman", Georgia, serif;
    text-align: center;
    color: #f2f4f8;
}

/* ===== BÖLÜM BAŞLIKLARI + MADDE ===== */
.section-title {
    position: relative;
    padding-left: 28px;
    margin-top: 25px;
    transition: transform 0.25s ease;
}

.section-title::before {
    content: "";
    position: absolute;
    left: 0;
    top: 50%;
    width: 14px;
    height: 14px;
    background-color: #6ec1ff; /* gökyüzü mavisi */
    border-radius: 50%;
    transform: translateY(-50%);
}

.section-title:hover {
    transform: scale(1.05);
}

/* ===== SLIDER (GERÇEKTEN KALIN) ===== */
div[data-baseweb="slider"] > div {
    height: 14px !important;
}

div[data-baseweb="slider"] > div > div {
    background-color: #6b1d1d !important; /* bordo */
    height: 14px !important;
}

div[data-baseweb="slider"] span {
    width: 24px !important;
    height: 24px !important;
    background-color: #8b2c2c !important;
}

/* ===== RADIO ===== */
div[role="radiogroup"] label {
    transform: scale(1.12);
    margin-right: 14px;
}

/* ===== BUTON ===== */
.stButton {
    display: flex;
    justify-content: center;
}

.stButton > button {
    background-color: #8b2c2c;
    color: #ffffff;
    border-radius: 22px;
    padding: 18px 80px;
    font-size: 24px;
    font-weight: 800;
    transition: transform 0.3s ease;
}

.stButton > button:hover {
    background-color: #6b1d1d;
    transform: scale(1.1);
}

/* ===== SONUÇ KARTI ===== */
.result-card {
    background: linear-gradient(135deg, #0b1d3a, #102a52);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.35);
}

/* ===== TABLO ===== */
.custom-table {
    background-color: rgba(10, 30, 60, 0.85);
    border-radius: 16px;
    padding: 20px;
    margin-top: 20px;
}

.custom-table th {
    color: #bcdcff;
}

.custom-table td {
    color: #f2f4f8;
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
    <p style="text-align:center; font-size:14px; opacity:0.9;">
    ⚠️ Bu sistem <b>tanı koymaz</b>. Klinik parametrelere dayalı
    <b>olasılıksal bir karar destek modeli</b> sunar.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# GİRDİLER
# =========================
st.markdown("<h3 class='section-title'>Demografik Bilgiler</h3>", unsafe_allow_html=True)
age = st.slider("Yaş", 1, 100, 50)
sex = st.radio("Cinsiyet", ["Female", "Male"], horizontal=True)

st.markdown("<h3 class='section-title'>Takip ve Tedavi Bilgileri</h3>", unsafe_allow_html=True)
n_days = st.slider("Takip Süresi (N_Days)", 0, 5000, 1000)
status = st.radio("Hasta Durumu (Status)", ["C", "CL", "D"], horizontal=True)
drug = st.radio("Uygulanan Tedavi (Drug)", ["Placebo", "D-penicillamine"], horizontal=True)

st.markdown("<h3 class='section-title'>Klinik Bulgular</h3>", unsafe_allow_html=True)
ascites = st.selectbox("Ascites", ["Yok", "Var"])
hepatomegaly = st.selectbox("Hepatomegaly", ["Yok", "Var"])
spiders = st.selectbox("Spiders", ["Yok", "Var"])
edema = st.selectbox("Edema", ["0", "1", "2"])

st.markdown("<h3 class='section-title'>Laboratuvar Bulguları</h3>", unsafe_allow_html=True)
bilirubin = st.slider("Bilirubin", 0.1, 30.0, 1.0)
cholesterol = st.slider("Cholesterol", 100.0, 500.0, 250.0)
albumin = st.slider("Albumin", 1.0, 6.0, 3.5)
copper = st.slider("Copper", 0.0, 300.0, 50.0)
alk_phos = st.slider("Alk_Phos", 50.0, 3000.0, 500.0)
sgot = st.slider("SGOT", 10.0, 500.0, 50.0)
trig = st.slider("Tryglicerides", 50.0, 500.0, 150.0)
platelets = st.slider("Platelets", 50.0, 500.0, 250.0)
prothrombin = st.slider("Prothrombin", 8.0, 20.0, 12.0)

# =========================
# BUTON
# =========================
predict_btn = st.button("EVRE TAHMİNİ YAP")
