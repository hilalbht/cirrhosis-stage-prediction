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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

.stApp {
    background: linear-gradient(135deg, #020024, #090979, #020024);
    color: #f2f4f8;
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, p, label {
    color: #f2f4f8 !important;
}

.stButton > button {
    background-color: #7b1e3a;
    color: #ffffff;
    border-radius: 16px;
    padding: 14px 50px;
    font-size: 22px;
    display: block;
    margin: 0 auto;
}

.stButton > button:hover {
    background-color: #5a162b;
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
🩺 Klinik Parametrelere Dayalı<br>Siroz Evre Tahmin Sistemi
</h1>
<p style="text-align:center;">
Eğitim ve klinik simülasyon amaçlı geliştirilmiştir.
</p>
""", unsafe_allow_html=True)

st.divider()

# =========================
# GİRDİLER
# =========================
age = st.slider("Yaş", 1, 100, 50)
sex = st.radio("Cinsiyet", ["Female", "Male"], horizontal=True)

n_days = st.slider("Takip Süresi (N_Days)", 0, 5000, 1000)
status = st.radio("Hasta Durumu", ["C", "CL", "D"], horizontal=True)
drug = st.radio("Tedavi", ["Placebo", "D-penicillamine"], horizontal=True)

ascites = st.selectbox("Ascites", ["Yok", "Var"])
hepatomegaly = st.selectbox("Hepatomegaly", ["Yok", "Var"])
spiders = st.selectbox("Spiders", ["Yok", "Var"])
edema = st.selectbox("Edema", ["0", "1", "2"])

bilirubin = st.slider("Bilirubin", 0.1, 30.0, 1.0)
albumin = st.slider("Albumin", 1.0, 6.0, 3.5)
platelets = st.slider("Platelets", 50.0, 500.0, 250.0)
prothrombin = st.slider("Prothrombin", 8.0, 20.0, 12.0)

st.divider()

# =========================
# TAHMİN
# =========================
if st.button("🔍 EVRE TAHMİNİ YAP"):

    input_df = pd.DataFrame([{
        "N_Days": n_days,
        "Status": {"C": 0, "CL": 1, "D": 2}[status],
        "Drug": 1 if drug == "D-penicillamine" else 0,
        "Age": age,
        "Sex": 1 if sex == "Male" else 0,
        "Ascites": 1 if ascites == "Var" else 0,
        "Hepatomegaly": 1 if hepatomegaly == "Var" else 0,
        "Spiders": 1 if spiders == "Var" else 0,
        "Edema": int(edema),
        "Bilirubin": bilirubin,
        "Albumin": albumin,
        "Platelets": platelets,
        "Prothrombin": prothrombin
    }])

    input_df = input_df[model.feature_names_in_]

    pred = model.predict(input_df)
    probs = model.predict_proba(input_df)[0]
    stage = le_stage.inverse_transform(pred)[0]

    # =========================
    # SONUÇ KARTI
    # =========================
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding:30px;
        border-radius:20px;
        text-align:center;">
        <h2>Tahmin Edilen Siroz Evresi</h2>
        <h1 style="color:#ffcccb;">Stage {stage}</h1>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📊 Evre Olasılıkları")
    for s, p in zip(le_stage.classes_, probs):
        st.progress(float(p), text=f"Stage {s}: %{p*100:.2f}")

    # =========================
    # MODEL KARAR MEKANİZMASI
    # =========================
    st.subheader("🧠 Bu Hasta Neden Bu Evrede?")

    base_proba = probs[np.argmax(probs)]
    impacts = []

    for col in input_df.columns:
        temp = input_df.copy()
        temp[col] = 0
        new_proba = model.predict_proba(temp)[0][np.argmax(probs)]
        impacts.append({
            "Parametre": col,
            "Etkisi": base_proba - new_proba
        })

    impact_df = pd.DataFrame(impacts).sort_values("Etkisi", ascending=False).head(5)

    st.dataframe(impact_df.style.format({"Etkisi": "{:.4f}"}))

    st.markdown("""
**Yorumlama:**
- Pozitif değer → parametre mevcut evreyi **güçlendiriyor**
- Negatif değer → parametre evreyi **zayıflatıyor**
- En üstteki parametreler modelin bu hasta için en belirleyici bulgularıdır
""")
