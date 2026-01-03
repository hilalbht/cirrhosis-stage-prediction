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
    background: #508194;
    background: radial-gradient(
        circle,
        rgba(80, 129, 148, 1) 0%,
        rgba(66, 168, 146, 1) 100%
    );
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
    color: #0f2a44;
    transition: transform 0.25s ease;
}
.section-title:hover {
    transform: scale(1.06);
}
.section-title::before {
    content: "● ";
    color: #0f2a44;
    font-weight: bold;
    font-size: 26px;
}
.stMarkdown h3.section-title {
    color: #0f2a44 !important;
}

/* ===== SECTION ALT AYIRICI ===== */
.section-divider {
    height: 1px;
    width: 100%;
    background: linear-gradient(
        to right,
        rgba(15, 42, 68, 0.9),
        rgba(15, 42, 68, 0.2)
    );
    margin: 6px 0 18px 0;
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

/* ===== INPUT KAYDIRMA ===== */
.section-title + div,
.section-title + .stSlider,
.section-title + .stRadio,
.section-title + .stSelectbox {
    margin-left: 100px;
}

/* ===== SLIDER RENGİ ===== */
/* ===== SLIDER TOP (THUMB) ===== */
div[data-baseweb="slider"] div[role="slider"] {
    background-color: #000000 !important;   /* top */
    border-color: #1f2933
 !important;
}

            /* ===== SLIDER VALUE (33 YAZISI) ===== */
div[data-baseweb="slider-value"] {
    color: #1f2933
 !important;
}

/* İçindeki span ihtimaline karşı */
div[data-baseweb="slider-value"] span {
    color: #1f2933
 !important;
}
div[data-baseweb="slider-value"] * {
    color: black !important;
}
/* ===== SLIDER VALUE ZORLA SİYAH ===== */
:root {
    --accent-color: #000000 !important;
}

div[data-baseweb="slider"] {
    --accent-color: #000000 !important;
}
/* ===== NUMBER INPUT ( + / - KONTROLLER ) ===== */

/* Dış kutu */
div[data-baseweb="input"] {
    background: rgba(15, 42, 68, 0.25) !important;
    border: 1.5px solid rgba(15, 42, 68, 0.45) !important;
    border-radius: 14px !important;
}

/* Yazı */
div[data-baseweb="input"] input {
    color: #0f2a44 !important;
    font-weight: 600;
    background: transparent !important;
}

[data-baseweb="input"]:focus-within {
    box-shadow: 0 0 0 2px rgba(66, 168, 146, 0.6) !important;
}


</style>
""", unsafe_allow_html=True)

# =========================
# MODEL
# =========================
model = joblib.load("xgboost_stage_model.pkl")
le_stage = joblib.load("stage_label_encoder.pkl")

# =========================
# HEADER
# =========================
st.markdown("""
<div style="background:rgba(15,42,68,.65);padding:30px;border-radius:22px;text-align:center">
<h1>Klinik Parametrelere Dayalı<br>Siroz Evre Tahmin Sistemi</h1>
<p><b>⚠️ Eğitim amaçlıdır</b></p>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================
# YARDIMCI FONKSİYON
# =========================
def slider_plus(key, label, minv, maxv, default, step=1, help=None):
    if key not in st.session_state:
        st.session_state[key] = default

    c1, c2 = st.columns([3,1])
    with c1:
        st.session_state[key] = st.slider(
            label, minv, maxv,
            st.session_state[key],
            step=step,
            label_visibility="collapsed",
            help=help
        )
    with c2:
        st.session_state[key] = st.number_input(
            " ", minv, maxv,
            st.session_state[key],
            step=step
        )
    return st.session_state[key]

# =========================
# DEMOGRAFİ
# =========================
st.subheader("Demografik Bilgiler")

age = slider_plus("age", "Yaş", 1, 100, 50)

sex = st.radio("Cinsiyet", ["Kadın", "Erkek"], horizontal=True)

st.divider()

# =========================
# TAKİP
# =========================
st.subheader("Takip ve Tedavi")

n_days = slider_plus(
    "n_days", "Takip Süresi (gün)",
    0, 5000, 1000, step=50,
    help="Tanıdan itibaren takip süresi"
)

status = st.radio(
    "Hasta Durumu",
    ["Stabil", "Komplikasyon gelişmiş", "Vefat"],
    horizontal=True
)

drug = st.radio(
    "Uygulanan Tedavi",
    ["Plasebo", "D-penisilamin"],
    horizontal=True
)

st.divider()

# =========================
# KLİNİK
# =========================
st.subheader("Klinik Bulgular")

ascites = st.selectbox("Karın İçi Sıvı Birikimi", ["Yok", "Var"])
hepatomegaly = st.selectbox("Karaciğer Büyümesi", ["Yok", "Var"])
spiders = st.selectbox("Örümcek Anjiyom", ["Yok", "Var"])
edema = st.selectbox("Ödem Düzeyi", ["0", "1", "2"])

st.divider()

# =========================
# LABORATUVAR (HEPSİ +/−)
# =========================
st.subheader("Laboratuvar Bulguları")

bilirubin   = slider_plus("bilirubin", "Bilirubin", 0.1, 30.0, 1.0, 0.1)
cholesterol = slider_plus("chol", "Kolesterol", 100.0, 500.0, 250.0, 10.0)
albumin     = slider_plus("albumin", "Albumin", 1.0, 6.0, 3.5, 0.1)
copper      = slider_plus("copper", "Serum Bakır", 0.0, 300.0, 50.0, 5.0)
alk_phos    = slider_plus("alk", "Alkalen Fosfataz", 50.0, 3000.0, 500.0, 50.0)
sgot        = slider_plus("sgot", "AST (SGOT)", 10.0, 500.0, 50.0, 5.0)
trig        = slider_plus("trig", "Trigliserid", 50.0, 500.0, 150.0, 10.0)
platelets   = slider_plus("plt", "Trombosit", 50.0, 500.0, 250.0, 10.0)
prothrombin = slider_plus("pt", "Protrombin", 8.0, 20.0, 12.0, 0.1)

st.divider()

# =========================
# BUTON
# =========================
predict_btn = st.button("EVRE TAHMİNİ YAP")

# =========================
# TAHMİN
# =========================
if predict_btn:
    input_df = pd.DataFrame([{
        "N_Days": n_days,
        "Status": {"Stabil":0,"Komplikasyon gelişmiş":1,"Vefat":2}[status],
        "Drug": 1 if drug=="D-penisilamin" else 0,
        "Age": age,
        "Sex": 1 if sex=="Erkek" else 0,
        "Ascites": 1 if ascites=="Var" else 0,
        "Hepatomegaly": 1 if hepatomegaly=="Var" else 0,
        "Spiders": 1 if spiders=="Var" else 0,
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
        "Status_label": {"Stabil":0,"Komplikasyon gelişmiş":1,"Vefat":2}[status],
        "Drug_label": 1 if drug=="D-penisilamin" else 0
    }])[model.feature_names_in_]

    probs = model.predict_proba(input_df)[0]
    stage = le_stage.inverse_transform([np.argmax(probs)])[0]

    st.success(f"Tahmin Edilen Evre: **Stage {stage}**")
