import streamlit as st
import pandas as pd
import joblib

# =========================
# SAYFA AYARLARI
# =========================
st.set_page_config(
    page_title="Siroz Evresi Tahmin Sistemi",
    layout="centered"
)

# Açık arka plan (göz yormaz)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# MODEL YÜKLE
# =========================
model = joblib.load("xgboost_stage_model.pkl")
le_stage = joblib.load("stage_label_encoder.pkl")

# =========================
# BAŞLIK
# =========================
st.title("🩺 Siroz Evresi Tahmin Sistemi")
st.write("Klinik ve laboratuvar değerlerini giriniz:")

st.divider()

# =========================
# KULLANICI GİRDİLERİ
# =========================

# Sayısal
age = st.slider("Yaş", 1, 100, 50)
n_days = st.slider("Takip Süresi (N_Days)", 0, 5000, 1000)

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

# Kategorik
sex = st.radio("Cinsiyet", ["Female", "Male"])
status = st.radio("Hasta Durumu (Status)", ["C", "D"])
drug = st.radio("İlaç (Drug)", ["Placebo", "D-penicillamine"])

ascites = st.selectbox("Ascites", ["Yok", "Var"])
hepatomegaly = st.selectbox("Hepatomegaly", ["Yok", "Var"])
spiders = st.selectbox("Spiders", ["Yok", "Var"])
edema = st.selectbox("Edema", ["0", "1", "2"])

# =========================
# TAHMİN BUTONU
# =========================
if st.button("🔍 Tahmin Et"):

    # Kategorik → sayısal
    sex_val = 1 if sex == "Male" else 0
    status_val = 1 if status == "D" else 0
    drug_val = 1 if drug == "D-penicillamine" else 0

    ascites_val = 1 if ascites == "Var" else 0
    hepatomegaly_val = 1 if hepatomegaly == "Var" else 0
    spiders_val = 1 if spiders == "Var" else 0
    edema_val = int(edema)

    # Model input
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
    }])

    # Sütun sırası
    input_df = input_df[model.feature_names_in_]

    # Tahmin
    pred = model.predict(input_df)
    probs = model.predict_proba(input_df)[0]

    stage = le_stage.inverse_transform(pred)[0]

    st.divider()

    # =========================
    # SONUÇ GÖSTERİMİ
    # =========================
    st.markdown(
        f"""
        <div style="
            background-color:#ffffff;
            padding:20px;
            border-radius:10px;
            box-shadow:0 0 10px rgba(0,0,0,0.05);
            text-align:center;
        ">
            <h2>🧬 Tahmin Edilen Siroz Evresi</h2>
            <h1 style="color:#1f77b4;">Stage {stage}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📊 Evre Olasılıkları")
    for s, p in zip(le_stage.classes_, probs):
        st.progress(float(p), text=f"Stage {s}: %{p*100:.2f}")
