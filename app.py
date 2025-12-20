import streamlit as st
import pandas as pd
import joblib

# Model ve encoder yükle
model = joblib.load("xgboost_stage_model.pkl")
le_stage = joblib.load("stage_label_encoder.pkl")

st.set_page_config(
    page_title="Siroz Evresi Tahmin Sistemi",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Siroz Evresi Tahmin Sistemi")
st.markdown("### Klinik karar destek & eğitim simülasyonu")

st.divider()

# ======================
# KULLANICI GİRDİLERİ
# ======================

st.subheader("🔬 Laboratuvar ve Klinik Değerler")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Yaş", 18, 90, 50)
    bilirubin = st.slider("Bilirubin", 0.1, 20.0, 1.2)
    albumin = st.slider("Albumin", 1.0, 5.5, 3.5)
    platelets = st.slider("Platelets", 50.0, 500.0, 250.0)

with col2:
    prothrombin = st.slider("Prothrombin", 8.0, 20.0, 12.0)
    edema = st.selectbox("Edema", [0, 1, 2])
    sex = st.radio("Cinsiyet", ["Kadın", "Erkek"])
    ascites = st.radio("Ascites", ["Yok", "Var"])

sex_val = 0 if sex == "Kadın" else 1
ascites_val = 0 if ascites == "Yok" else 1

st.divider()

# ======================
# TAHMİN BUTONU
# ======================

if st.button("🧠 Tahmin Et", use_container_width=True):

    # Modelin beklediği tüm feature'ları doldur
    input_df = pd.DataFrame([{
        "N_Days": 0,
        "Status": 0,
        "Drug": 0,
        "Age": age,
        "Sex": sex_val,
        "Ascites": ascites_val,
        "Hepatomegaly": 0,
        "Spiders": 0,
        "Edema": edema,
        "Bilirubin": bilirubin,
        "Cholesterol": 0,
        "Albumin": albumin,
        "Copper": 0,
        "Alk_Phos": 0,
        "SGOT": 0,
        "Tryglicerides": 0,
        "Platelets": platelets,
        "Prothrombin": prothrombin,
        "Status_label": 0,
        "Drug_label": 0
    }])

    input_df = input_df[model.feature_names_in_]

    pred = model.predict(input_df)
    probs = model.predict_proba(input_df)[0]

    stage = le_stage.inverse_transform(pred)[0]

    st.success(f"🎯 **Tahmin Edilen Siroz Evresi: Stage {stage}**")

    st.subheader("📊 Evre Olasılıkları")
    for s, p in zip(le_stage.classes_, probs):
        st.progress(float(p), text=f"Stage {s}: %{p*100:.2f}")
