import streamlit as st
import pandas as pd
import joblib

# Model yükle
model = joblib.load("xgboost_stage_model.pkl")
le_stage = joblib.load("stage_label_encoder.pkl")

st.title("🩺 Siroz Evresi Tahmin Sistemi")

st.write("Laboratuvar değerlerini giriniz.")

bilirubin = st.slider("Bilirubin", 0.1, 20.0, 1.0)
albumin = st.slider("Albumin", 1.0, 5.0, 3.5)
platelets = st.slider("Platelets", 50.0, 500.0, 250.0)
prothrombin = st.slider("Prothrombin", 8.0, 20.0, 12.0)
edema = st.selectbox("Edema", [0, 1, 2])

if st.button("Tahmin Et"):
    input_df = pd.DataFrame([{
        "Bilirubin": bilirubin,
        "Albumin": albumin,
        "Platelets": platelets,
        "Prothrombin": prothrombin,
        "Edema": edema
    }])

    # Eksik sütunları tamamla
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model.feature_names_in_]

    pred = model.predict(input_df)
    stage = le_stage.inverse_transform(pred)[0]

    probs = model.predict_proba(input_df)[0]

    st.success(f"Tahmin Edilen Siroz Evresi: Stage {stage}")

    for s, p in zip(le_stage.classes_, probs):
        st.write(f"Stage {s}: %{p*100:.2f}")
