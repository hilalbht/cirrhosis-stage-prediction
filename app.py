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
div[data-baseweb="slider"] {
    --accent-color: #0f2a44;
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

# --- Demografik Bilgiler ---
st.subheader("DEMOGRAFİK BİLGİLER")

age = st.slider("Yaş (1-100)", 1, 100, 50)

sex_map = {"Kadın": 0, "Erkek": 1}
sex_input = st.radio("Cinsiyet", list(sex_map.keys()), horizontal=True)
sex_val = sex_map[sex_input]

st.divider()

# --- Takip ve Tedavi ---
st.subheader("TAKİP ve TEDAVİ BİLGİLERİ")

n_days = st.slider("Takip Süresi (Gün)", 0, 5000, 1000)

status_map = {
    "Tam Fonksiyonel / Sağlıklı": 0,
    "Kısmen Sağ / Fonksiyonel": 1,
    "Kaybedilmiş / Fonksiyon Kaybı": 2
}
status_input = st.radio("HASTA DURUMU", list(status_map.keys()), horizontal=True)
status_val = status_map[status_input]

drug_map = {"Plasebo":0, "D-penisilamin":1}
drug_input = st.radio("Uygulanan Tedavi", list(drug_map.keys()), horizontal=True)
drug_val = drug_map[drug_input]

st.divider()

# --- Klinik Bulgular ---
st.subheader("FİZİKSEL BULGULAR")

ascites_map = {"Karın Boşluğunda Sıvı Birikimi Yok()":0, "Karın Boşluğunda Sıvı Birikimi Var )":1}
ascites_input = st.selectbox("Ascites (Asit)", list(ascites_map.keys()))
ascites_val = ascites_map[ascites_input]

hepatomegaly_map = {"Karaciğer Büyümesi Yok":0, "Karaciğer Büyümesi Var":1}
hepatomegaly_input = st.selectbox("Hepatomegaly", list(hepatomegaly_map.keys()))
hepatomegaly_val = hepatomegaly_map[hepatomegaly_input]

spiders_map = {"Ciltte Örümcek Damarlar Şeklinde Genişleme Yok":0, "Ciltte Örümcek Damarlar Şeklinde Genişleme Var )":1}
spiders_input = st.selectbox("Spiders (Örümcek damarlar)", list(spiders_map.keys()))
spiders_val = spiders_map[spiders_input]

edema_map = {
    "Az (hafif ödem)": 0,
    "Orta (orta seviyede ödem)": 1,
    "Şiddetli (yaygın vücut ödemi)": 2
}
edema_input = st.selectbox("Ödem", list(edema_map.keys()))
edema_val = edema_map[edema_input]

st.divider()

# --- Laboratuvar Bulguları ---
st.subheader("TEST SONUÇLARI")

bilirubin = st.slider("Bilirubin (mg/dL)", 0.1, 30.0, 1.0)
cholesterol = st.slider("Cholesterol (mg/dL)", 100.0, 500.0, 250.0)
albumin = st.slider("Albumin (g/dL)", 1.0, 6.0, 3.5)
copper = st.slider("Copper (µg/dL)", 0.0, 300.0, 50.0)
alk_phos = st.slider("Alk_Phos (IU/L)", 50.0, 3000.0, 500.0)
sgot = st.slider("SGOT (IU/L)", 10.0, 500.0, 50.0)
trig = st.slider("Tryglicerides (mg/dL)", 50.0, 500.0, 150.0)
platelets = st.slider("Platelets (10^3/µL)", 50.0, 500.0, 250.0)
prothrombin = st.slider("Prothrombin (%)", 8.0, 20.0, 12.0)

st.divider()

# =========================
# TAHMİN BUTONU
# =========================
if st.button("EVRE TAHMİNİ YAP"):
    
    # Input dataframe oluştur
    input_df = pd.DataFrame([{
        "N_Days": n_days,
        "Status": status_val,
        "Drug": drug_val,
        "Age": age,
        "Sex": sex_val,
        "Ascites": ascites_val,
        "Hepatomegaly": hepatomegaly_val,
        "Spiders": spiders_val,
        "Edema": edema_val,
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

    st.subheader("EVRE OLASILIKLARI")
    for s, p in zip(le_stage.classes_, probs):
        st.progress(float(p), text=f"Stage {s}: %{p*100:.2f}")

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    st.subheader("⚠️ HASTA BAZLI PARAMETRE ETKİ ANALİZİ")
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
