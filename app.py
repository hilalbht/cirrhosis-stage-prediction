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
    border-color: #000000 !important;
}

/* ===== SLIDER ÜSTÜNDEKİ SAYI ===== */
div[data-baseweb="slider"] div[role="slider"] span {
    color: #000000 !important;   /* sayı */
}

/* ===== HOVER / ACTIVE DURUMU ===== */
div[data-baseweb="slider"] div[role="slider"]:hover {
    background-color: #000000 !important;
}
            /* ===== SLIDER VALUE (33 YAZISI) ===== */
div[data-baseweb="slider-value"] {
    color: #000000 !important;
}

/* İçindeki span ihtimaline karşı */
div[data-baseweb="slider-value"] span {
    color: #000000 !important;
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
    <h1>Klinik Parametrelere Dayalı<br>Siroz Evre Tahmin Sistemi</h1>
    <p><b>⚠️Eğitim ve klinik simülasyon amaçlıdır.</b></p>
    <p style="font-size:14px;">
        ⚠️Bu sistem <b>olasılıksal ve istatistiksel bir tahmin</b> üretir.  
        Klinik kararların yerine geçmez.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================
# GİRDİLER
# =========================
st.markdown("<h3 class='section-title'>Demografik Bilgiler</h3>", unsafe_allow_html=True)
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# Age
age = st.slider("Yaş", 1, 100, 50)

# Sex
# Sex: 0 = Kadın, 1 = Erkek
sex = st.radio(
    "Cinsiyet",
    ["Kadın", "Erkek"],
    horizontal=True
)

st.divider()

st.markdown("<h3 class='section-title'>Takip ve Tedavi Bilgileri</h3>", unsafe_allow_html=True)
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# N_Days: Hastanın takip süresi (gün)
n_days = st.slider(
    "Takip Süresi (gün)",
    0, 5000, 1000,
    help="Hastanın tanıdan itibaren klinik olarak takip edildiği toplam süreyi ifade eder."
)

# Status
# C  -> Stabil
# CL -> Komplikasyon gelişmiş
# D  -> Vefat
status = st.radio(
    "Hasta Durumu",
    ["Stabil", "Komplikasyon gelişmiş", "Vefat"],
    help="Hastanın klinik takip sürecindeki genel durumunu ifade eder.",
    horizontal=True
)

# Drug
# Drug: 0 = Placebo, 1 = D-penicillamine
drug = st.radio(
    "Uygulanan Tedavi",
    ["Plasebo", "D-penisilamin"],
    horizontal=True
)

st.divider()

st.markdown("<h3 class='section-title'>Klinik Bulgular</h3>", unsafe_allow_html=True)
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# Ascites: Karın içi sıvı birikimi
ascites = st.selectbox(
    "Karın İçi Sıvı Birikimi",
    ["Yok", "Var"],
    help="Karın boşluğunda sıvı birikmesi durumudur."
)

# Hepatomegaly: Karaciğer büyümesi
hepatomegaly = st.selectbox(
    "Karaciğer Büyümesi",
    ["Yok", "Var"],
    help="Karaciğerin normal boyutlarının üzerine çıkması durumudur."
)

# Spiders: Örümcek anjiyom
spiders = st.selectbox(
    "Örümcek Anjiyom",
    ["Yok", "Var"],
    help="Cilt yüzeyinde görülen kılcal damar genişlemeleridir."
)

# Edema: Ödem derecesi (0-2)
edema = st.selectbox(
    "Ödem Düzeyi",
    ["0", "1", "2"],
    help="0: Yok, 1: Hafif, 2: Belirgin ödem"
)

st.divider()

st.markdown("<h3 class='section-title'>Laboratuvar Bulguları</h3>", unsafe_allow_html=True)
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# Laboratuvar parametreleri
bilirubin = st.slider("Bilirubin", 0.1, 30.0, 1.0)
cholesterol = st.slider("Kolesterol", 100.0, 500.0, 250.0)
albumin = st.slider("Albumin", 1.0, 6.0, 3.5)
copper = st.slider("Serum Bakır Düzeyi", 0.0, 300.0, 50.0)
alk_phos = st.slider("Alkalen Fosfataz", 50.0, 3000.0, 500.0)
sgot = st.slider("AST (SGOT)", 10.0, 500.0, 50.0)
trig = st.slider("Trigliserid", 50.0, 500.0, 150.0)
platelets = st.slider("Trombosit Sayısı", 50.0, 500.0, 250.0)
prothrombin = st.slider("Protrombin Zamanı", 8.0, 20.0, 12.0)

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
    sex_val = 1 if sex == "Erkek" else 0

    status_val = {
        "Stabil": 0,
        "Komplikasyon gelişmiş": 1,
        "Vefat": 2
    }[status]

    drug_val = 1 if drug == "D-penisilamin" else 0

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

    probs = model.predict_proba(input_df)[0]
    stage = le_stage.inverse_transform([np.argmax(probs)])[0]

    st.markdown(f"""
    <div class="result-card">
        <h2>Tahmin Edilen Siroz Evresi</h2>
        <h1 style="font-size:48px;">Stage {stage}</h1>
        <p style="font-size:14px;">
        Bu çıktı, modelin mevcut verilere dayanarak yaptığı <b>istatistiksel bir tahmindir</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Evre Olasılıkları")
    for s, p in zip(le_stage.classes_, probs):
        st.progress(float(p), text=f"Stage {s}: %{p*100:.2f}")

    st.markdown("<br><br><br>", unsafe_allow_html=True)

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
