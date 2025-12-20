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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

.stApp {
    background: linear-gradient(90deg,
        rgba(88,88,112,1) 28%,
        rgba(66,66,179,1) 68%,
        rgba(0,212,255,1) 100%);
    color: #f2f4f8;
    font-family: 'Inter', sans-serif;
}

/* ===== ANA BAŞLIK KARTI ===== */
.header-card {
    background: rgba(15, 42, 68, 0.65);
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.35);
    transition: transform 0.35s ease;
}
.header-card:hover {
    transform: scale(1.06);
}

/* ===== BÖLÜM BAŞLIKLARI ===== */
.section-title {
    font-family: "Times New Roman", Georgia, serif;
    font-size: 26px;
    transition: transform 0.25s ease;
}
.section-title:hover {
    transform: scale(1.06);
}
.section-title::before {
    content: "● ";
    color: #111827;
    font-weight: bold;
    font-size: 26px;
}

/* ===== YENİ: HAFİF SECTION KART ===== */
.section-card {
    background: rgba(15, 42, 68, 0.35);
    border-radius: 18px;
    padding: 22px 26px;
    margin-top: 12px;
    box-shadow: 0px 8px 22px rgba(0,0,0,0.25);
    transition: transform 0.25s ease;
}
.section-card:hover {
    transform: scale(1.03);
}

/* ===== BUTON ===== */
.stButton > button {
    background: linear-gradient(135deg, #111827, #1f2933);
    color: #ffffff;
    border-radius: 22px;
    padding: 26px 110px;
    font-size: 30px;
    font-weight: 900;
    letter-spacing: 2px;
    box-shadow: 0px 12px 40px rgba(0,0,0,0.45);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: scale(1.18);
}

/* ===== SONUÇ KARTI ===== */
.result-card {
    background: linear-gradient(135deg, #0f2a44, #123a5f);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
}

/* ===== TABLO ===== */
.custom-table {
    background-color: rgba(15, 42, 68, 0.85);
    border-radius: 16px;
    padding: 20px;
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
    <h1>"Klinik Parametrelere Dayalı<br>Siroz Evre Tahmin Sistemi"</h1>
    <p><b>⚠️Eğitim ve klinik simülasyon amaçlıdır.</b></p>
    <p style="font-size:14px;">
        ⚠️Bu sistem <b>olasılıksal ve istatistiksel bir tahmin</b> üretir.  
        Klinik kararların yerine geçmez.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================
# DEMOGRAFİK
# =========================
st.markdown("<h3 class='section-title'>Demografik Bilgiler</h3>", unsafe_allow_html=True)
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
age = st.slider("Yaş", 1, 100, 50)
sex = st.radio("Cinsiyet", ["Female", "Male"], horizontal=True)
st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# =========================
# TAKİP / TEDAVİ
# =========================
st.markdown("<h3 class='section-title'>Takip ve Tedavi Bilgileri</h3>", unsafe_allow_html=True)
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
n_days = st.slider("Takip Süresi (N_Days)", 0, 5000, 1000)
status = st.radio("Hasta Durumu (Status)", ["C", "CL", "D"], horizontal=True)
drug = st.radio("Uygulanan Tedavi (Drug)", ["Placebo", "D-penicillamine"], horizontal=True)
st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# =========================
# KLİNİK BULGULAR
# =========================
st.markdown("<h3 class='section-title'>Klinik Bulgular</h3>", unsafe_allow_html=True)
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
ascites = st.selectbox("Ascites", ["Yok", "Var"])
hepatomegaly = st.selectbox("Hepatomegaly", ["Yok", "Var"])
spiders = st.selectbox("Spiders (Örümcek anjiyom)", ["Yok", "Var"])
edema = st.selectbox("Edema", ["0", "1", "2"])
st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# =========================
# LAB
# =========================
st.markdown("<h3 class='section-title'>Laboratuvar Bulguları</h3>", unsafe_allow_html=True)
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
bilirubin = st.slider("Bilirubin", 0.1, 30.0, 1.0)
cholesterol = st.slider("Cholesterol", 100.0, 500.0, 250.0)
albumin = st.slider("Albumin", 1.0, 6.0, 3.5)
copper = st.slider("Copper", 0.0, 300.0, 50.0)
alk_phos = st.slider("Alk_Phos", 50.0, 3000.0, 500.0)
sgot = st.slider("SGOT", 10.0, 500.0, 50.0)
trig = st.slider("Tryglicerides", 50.0, 500.0, 150.0)
platelets = st.slider("Platelets", 50.0, 500.0, 250.0)
prothrombin = st.slider("Prothrombin", 8.0, 20.0, 12.0)
st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# =========================
# BUTON
# =========================
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_btn = st.button("EVRE TAHMİNİ YAP")