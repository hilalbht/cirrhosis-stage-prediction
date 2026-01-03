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
# STİL (CSS) — AYNI
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
.stApp {
    background: radial-gradient(circle, rgba(80,129,148,1), rgba(66,168,146,1));
    font-family: 'Inter', sans-serif;
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
