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
    transition: transform 0.3s ease;
}
.header-card:hover {
    transform: scale(1.04);
}

/* ===== ANA BAŞLIK ===== */
.main-title {
    font-family: "Times New Roman", Georgia, serif;
    margin-bottom: 10px;
}

/* ===== BÖLÜM BAŞLIKLARI ===== */
.section-title {
    transition: transform 0.25s ease;
}
.section-title:hover {
    transform: scale(1.05);
}
.section-title::before {
    content: "● ";
    color: #5dade2;
    font-weight: bold;
}

/* ===== SLIDER (YEŞİL) ===== */
div[data-baseweb="slider"] > div > div {
    background-color: #2ecc71 !important;
}

/* ===== BUTON ===== */
.stButton > button {
    background-color: #1f6fb2;
    color: #ffffff;
    border-radius: 18px;
    padding: 16px 70px;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: 1px;
    transition: transform 0.3s ease;
}
.stButton > button:hover {
    background-color: #164f82;
    transform: scale(1.12);
}

/* ===== SONUÇ KARTI ===== */
.result-card {
    background: linear-gradient(135deg, #0f2a44, #123a5f);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.25);
    transition: transform 0.3s ease;
}
.result-card:hover {
    transform: scale(1.06);
}

/* ===== TABLO ===== */
.custom-table {
    background-color: rgba(15, 42, 68, 0.85);
    border-radius: 16px;
    padding: 20px;
    margin-top: 25px;
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
    <p>
        Eğitim ve klinik simülasyon amaçlı geliştirilmiştir.
    </p>
    <p style="font-size:14px;">
        Bu sistem, klinik parametrelere dayalı <b>olasılıksal bir evre tahmini</b> sunar.<br>
        Sonuçlar <b>tanısal doğruluk garantisi içermez</b> ve klinik kararların yerine geçmez.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================
# GİRDİLER
# =========================
st.markdown("<h3 class='section-title'>Demografik Bilgiler</h3>", unsafe_allow_html=True)
age = st.slider("Yaş", 1, 100, 50)
sex = st.radio("Cinsiyet", ["Female", "Male"], horizontal=True)

st.divider()

st.markdown("<h3 class='section-title'>Takip ve Tedavi Bilgileri</h3>", unsafe_allow_html=True)
n_days = st.slider("Takip Süresi (N_Days)", 0, 5000, 1000)
status = st.radio("Hasta Durumu (Status)", ["C", "CL", "D"], horizontal=True)
drug = st.radio("Uygulanan Tedavi (Drug)", ["Placebo", "D-penicillamine"], horizontal=True)

st.divider()

st.markdown("<h3 class='section-title'>Klinik Bulgular</h3>", unsafe_allow_html=True)
ascites = st.selectbox("Ascites", ["Yok", "Var"])
hepatomegaly = st.selectbox("Hepatomegaly", ["Yok", "Var"])
spiders = st.selectbox("Spiders", ["Yok", "Var"])
edema = st.selectbox("Edema", ["0", "1", "2"])

st.divider()

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

st.divider()

# =========================
# BUTON
# =========================
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_btn = st.button("🔍 EVRE TAHMİNİ YAP")

# =========================
# TAHMİN
# =========================
if predict_btn:
    sex_val = 1 if sex == "Male" else 0
    status_val = {"C":0,"CL":1,"D":2}[status]
    drug_val = 1 if drug == "D-penicillamine" else 0

    input_df = pd.DataFrame([{
        "N_Days": n_days,
        "Status": status_val,
        "Drug": drug_val,
        "Age": age,
        "Sex": sex_val,
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
        "Status_label": status_val,
        "Drug_label": drug_val
    }])[model.feature_names_in_]

    probs = model.predict_proba(input_df)[0]
    stage = le_stage.inverse_transform([np.argmax(probs)])[0]

    st.markdown(f"""
    <div class="result-card">
        <h2>Tahmin Edilen Siroz Evresi</h2>
        <h1 style="font-size:48px;">Stage {stage}</h1>
        <p style="font-size:14px;">
        Not: Gösterilen evre, modelin mevcut verilere dayanarak yaptığı
        <b>istatistiksel bir tahmindir</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # =========================
    # HASTA BAZLI PARAMETRE ANALİZİ
    # =========================
    st.subheader("⚠️ Hasta Bazlı Parametre Etki Analizi")
    st.write(
        "Aşağıda, modelin **bu hasta için** tahmin edilen evreye "
        "en fazla katkı sağlayan klinik parametreler gösterilmektedir."
    )

    base_index = np.argmax(probs)
    impact_results = []

    for col in model.feature_names_in_:
        temp_df = input_df.copy()
        temp_df[col] = 0
        temp_proba = model.predict_proba(temp_df)[0]
        diff = probs[base_index] - temp_proba[base_index]

        if diff > 0:
            yorum = "Bu parametre evreyi artırıyor / risk oluşturuyor."
        elif diff < 0:
            yorum = "Evre tahminini azaltıcı yönde etkili."
        else:
            yorum = "Belirgin etkisi yok."

        impact_results.append({
            "Parametre": col,
            "Etki Büyüklüğü": diff,
            "Klinik Yorum": yorum
        })

    impact_df = pd.DataFrame(impact_results)\
        .sort_values("Etki Büyüklüğü", ascending=False)\
        .head(5)

    st.markdown(f"""
    <div class="custom-table">
        {impact_df.to_html(index=False)}
    </div>
    """, unsafe_allow_html=True)
