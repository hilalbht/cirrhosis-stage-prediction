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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@500;600&display=swap');

.stApp {
    background: linear-gradient(90deg,
        rgba(2, 0, 36, 1) 0%,
        rgba(9, 9, 121, 1) 51%,
        rgba(0, 212, 255, 1) 100%);
    color: #f2f4f8;
    font-family: 'Inter', sans-serif;
}

h1 {
    font-family: 'Playfair Display', serif;
    color: #f2f4f8 !important;
}

h2, h3, label, p {
    color: #f2f4f8 !important;
}

/* Madde işaretleri */
h3::before {
    content: "● ";
    color: #5dade2;
    font-weight: bold;
}

/* Slider rengi */
div[data-baseweb="slider"] > div > div {
    background-color: #1f6fb2 !important;
}

/* Buton */
.stButton > button {
    background-color: #1f6fb2;
    color: #ffffff;
    border-radius: 16px;
    padding: 14px 60px;
    font-size: 22px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #164f82;
}
</style>
""", unsafe_allow_html=True)

# =========================
# MODEL YÜKLE
# =========================
model = joblib.load("xgboost_stage_model.pkl")
le_stage = joblib.load("stage_label_encoder.pkl")

# =========================
# BAŞLIK
# =========================
st.markdown("""
<h1 style="text-align:center;">
 Klinik Parametrelere Dayalı<br>Siroz Evre Tahmin Sistemi
</h1>
<p style="text-align:center;">
Eğitim ve klinik simülasyon amaçlı geliştirilmiştir.
</p>
""", unsafe_allow_html=True)

st.divider()

# =========================
# GİRDİLER
# =========================
st.markdown("<h3>Demografik Bilgiler</h3>", unsafe_allow_html=True)
age = st.slider("Yaş", 1, 100, 50)
sex = st.radio("Cinsiyet", ["Female", "Male"], horizontal=True)

st.divider()

st.markdown("<h3>Takip ve Tedavi Bilgileri</h3>", unsafe_allow_html=True)
n_days = st.slider("Takip Süresi (N_Days)", 0, 5000, 1000)
status = st.radio("Hasta Durumu (Status)", ["C", "CL", "D"], horizontal=True)
drug = st.radio("Uygulanan Tedavi (Drug)", ["Placebo", "D-penicillamine"], horizontal=True)

st.divider()

st.markdown("<h3>Klinik Bulgular</h3>", unsafe_allow_html=True)
ascites = st.selectbox("Ascites (Karın içi sıvı birikimi)", ["Yok", "Var"])
hepatomegaly = st.selectbox("Hepatomegaly (Karaciğer büyümesi)", ["Yok", "Var"])
spiders = st.selectbox("Spiders (Örümcek anjiyom)", ["Yok", "Var"])
edema = st.selectbox("Edema (Ödem durumu)", ["0", "1", "2"])

st.divider()

st.markdown("<h3>Laboratuvar Bulguları</h3>", unsafe_allow_html=True)
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
# BUTON (ORTALANMIŞ)
# =========================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔍 EVRE TAHMİNİ YAP")

# =========================
# TAHMİN
# =========================
if predict_btn:

    sex_val = 1 if sex == "Male" else 0
    status_map = {"C": 0, "CL": 1, "D": 2}
    status_val = status_map[status]
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
    }])

    input_df = input_df[model.feature_names_in_]

    pred = model.predict(input_df)
    probs = model.predict_proba(input_df)[0]
    stage = le_stage.inverse_transform(pred)[0]

    st.divider()

    # =========================
    # SONUÇ KARTI
    # =========================
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #0f2a44, #123a5f);
        padding:30px;
        border-radius:18px;
        text-align:center;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.25);
    ">
        <h2 style="color:#dcefff;">Tahmin Edilen Siroz Evresi</h2>
        <h1 style="color:#ffffff; font-size:48px;">Stage {stage}</h1>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Evre Olasılıkları")
    for s, p in zip(le_stage.classes_, probs):
        st.progress(float(p), text=f"Stage {s}: %{p*100:.2f}")

    st.subheader("Karar Mekanizmasını Etkileyen Baskın Faktörler")
    fi_df = pd.DataFrame({
        "Özellik": model.feature_names_in_,
        "Önem": model.feature_importances_
    }).sort_values(by="Önem", ascending=False).head(10)
    st.bar_chart(fi_df.set_index("Özellik"))

    # =========================
    # KİŞİYE ÖZEL RİSK ANALİZİ (RENKLİ)
    # =========================
    st.subheader("⚠️ Hasta Bazlı Parametre Etki Analizi")
    st.subheader(" Bu Hasta Neden Bu Evrede?")

    base_proba = model.predict_proba(input_df)[0]
    base_stage_index = np.argmax(base_proba)

    impact_results = []
    for col in model.feature_names_in_:
        temp_df = input_df.copy()
        temp_df[col] = 0
        temp_proba = model.predict_proba(temp_df)[0]
        diff = base_proba[base_stage_index] - temp_proba[base_stage_index]

        if diff > 0:
            yorum = "Bu parametre evreyi artırıyor / risk oluşturuyor"
            color = 'color:red; font-weight:bold;'
        elif diff < 0:
            yorum = "Bu parametre evreyi düşürüyor / koruyucu etki"
            color = 'color:green; font-weight:bold;'
        else:
            yorum = "Etkisi yok"
            color = 'color:gray;'

        impact_results.append({
            "Parametre": f"<span style='{color}'>{col}</span>",
            "Etkisi": diff,
            "Yorum": yorum
        })

    impact_df = pd.DataFrame(impact_results).sort_values(by="Etkisi", ascending=False).head(5)
    st.write(impact_df.to_html(escape=False, index=False), unsafe_allow_html=True)
