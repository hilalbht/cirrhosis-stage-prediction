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
.stApp {
    background: linear-gradient(90deg,
        rgba(0, 66, 97, 1) 0%,
        rgba(87, 199, 133, 1) 50%,
        rgba(237, 221, 83, 1) 100%);
    color: #f5f5f5;
}

h1, h2, h3, label, p {
    color: #f5f5f5 !important;
}

div[data-baseweb="slider"] > div > div {
    background-color: #7b1e3a !important;
}

.stButton > button {
    background-color: #7b1e3a;
    color: #f5f5f5;
    border-radius: 12px;
    padding: 12px 40px;
    font-size: 20px;
    display: block;
    margin: 0 auto;
}

.stButton > button:hover {
    background-color: #5a162b;
    color: #ffffff;
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
st.markdown("<h1 style='text-align:center;'>🩺 Klinik Parametrelere Dayalı<br>Siroz Evre Tahmin Sistemi</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Eğitim ve klinik simülasyon amaçlı geliştirilmiştir.</p>", unsafe_allow_html=True)

st.divider()

# =========================
# GİRDİLER
# =========================
# =========================
# GİRDİLER
# =========================

# --- DEMOGRAFİK BİLGİLER ---
st.subheader(" Demografik Bilgiler")

age = st.slider("Yaş", 1, 100, 50)
sex = st.radio("Cinsiyet", ["Female", "Male"], horizontal=True)

st.divider()

# --- TAKİP ve TEDAVİ BİLGİLERİ ---
st.subheader(" Takip ve Tedavi Bilgileri")

n_days = st.slider("Takip Süresi (N_Days)", 0, 5000, 1000)
status = st.radio("Hasta Durumu (Status)", ["C", "CL", "D"], horizontal=True)
drug = st.radio("Uygulanan Tedavi (Drug)", ["Placebo", "D-penicillamine"], horizontal=True)

st.divider()

# --- KLİNİK BULGULAR ---
st.subheader(" Klinik Bulgular")

ascites = st.selectbox("Ascites", ["Yok", "Var"])
hepatomegaly = st.selectbox("Hepatomegaly", ["Yok", "Var"])
spiders = st.selectbox("Spiders", ["Yok", "Var"])
edema = st.selectbox("Edema", ["0", "1", "2"])

st.divider()

# --- LABORATUVAR BULGULARI ---
st.subheader(" Laboratuvar Bulguları")

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
# TAHMİN
# =========================
if st.button("EVRE TAHMİNİ YAP"):

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

    st.markdown(f"""
    <div style="background-color:#ffffff;
                padding:25px;
                border-radius:15px;
                text-align:center;">
        <h2 style="color:#004261;">Tahmin Edilen Siroz Evresi</h2>
        <h1 style="color:#7b1e3a;">Stage {stage}</h1>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Evre Olasılıkları")
    for s, p in zip(le_stage.classes_, probs):
        st.progress(float(p), text=f"Stage {s}: %{p*100:.2f}")

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    st.subheader("Modelin Karar Mekanizması (Feature Importance)")

    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Özellik": model.feature_names_in_,
        "Önem": importances
    }).sort_values(by="Önem", ascending=False).head(10)

    st.bar_chart(fi_df.set_index("Özellik"))
    # =========================
    # KİŞİYE ÖZEL RİSK ANALİZİ
    # =========================
    st.subheader("⚠️ Hasta Bazlı Parametre Etki Analizi")

    base_proba = model.predict_proba(input_df)[0]
    base_stage_index = np.argmax(base_proba)

    impact_results = []

    for col in model.feature_names_in_:
        temp_df = input_df.copy()
        temp_df[col] = 0  # parametreyi nötrle

        temp_proba = model.predict_proba(temp_df)[0]
        diff = base_proba[base_stage_index] - temp_proba[base_stage_index]

        impact_results.append({
            "Parametre": col,
            "Etkisi": diff
        })

    impact_df = pd.DataFrame(impact_results)
    impact_df = impact_df.sort_values(by="Etkisi", ascending=False).head(5)

    st.write(
        "Aşağıda, **bu hasta için** evre tahminini en fazla etkileyen klinik parametreler yer almaktadır:"
    )

    st.dataframe(
        impact_df.style.format({"Etkisi": "{:.4f}"})
    )
