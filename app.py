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

st.markdown(
    """
    <style>
    /* Arka plan */
    .stApp {
        background-color: #f7f9fc;
        color: #1a1a1a;
    }

    /* Genel yazılar */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #1a1a1a !important;
    }

    /* Slider ve input yazıları */
    .stSlider label,
    .stRadio label,
    .stSelectbox label {
        color: #1a1a1a !important;
    }

    /* Progress bar yazıları */
    .stProgress > div > div > div > div {
        color: #1a1a1a !important;
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
st.caption(
    "Bu sistem, hastaya ait klinik ve laboratuvar verilerini kullanarak "
    "siroz hastalığının evresini (Stage) makine öğrenmesi ile tahmin eder."
)

st.divider()

# =========================
# DEMOGRAFİK BİLGİLER
# =========================
st.subheader("👤 Demografik Bilgiler")
st.caption("Hastaya ait temel bilgiler")

age = st.slider("Yaş", 1, 100, 50)
sex = st.radio("Cinsiyet", ["Female", "Male"])

st.divider()

# =========================
# TAKİP & TEDAVİ
# =========================
st.subheader("📅 Takip ve Tedavi Bilgileri")
st.caption("Hastanın izlem süresi ve aldığı tedavi")

n_days = st.slider(
    "Takip Süresi (Gün)",
    0, 5000, 1000,
    help="Hastanın çalışmaya dahil edildiği günden itibaren takip süresi"
)

status = st.radio(
    "Hasta Durumu (Status)",
    ["C", "D"],
    help="C: Yaşıyor, D: Vefat etmiş"
)

drug = st.radio(
    "Kullanılan İlaç",
    ["Placebo", "D-penicillamine"],
    help="Hastanın aldığı tedavi türü"
)

st.divider()

# =========================
# KLİNİK BULGULAR
# =========================
st.subheader("🧬 Klinik Bulgular")
st.caption("Fiziksel muayene ve gözleme dayalı bulgular")

ascites = st.selectbox(
    "Ascites (Karında Sıvı Birikimi)",
    ["Yok", "Var"]
)

hepatomegaly = st.selectbox(
    "Hepatomegaly (Karaciğer Büyümesi)",
    ["Yok", "Var"]
)

spiders = st.selectbox(
    "Spiders (Örümcek Anjiomları)",
    ["Yok", "Var"]
)

edema = st.selectbox(
    "Edema (Ödem Seviyesi)",
    ["0", "1", "2"],
    help="0: Yok, 1: Hafif, 2: Şiddetli"
)

st.divider()

# =========================
# LABORATUVAR DEĞERLERİ
# =========================
st.subheader("🧪 Laboratuvar Bulguları")
st.caption("Kan testlerinden elde edilen biyokimyasal değerler")

bilirubin = st.slider("Bilirubin (mg/dL)", 0.1, 30.0, 1.0)
cholesterol = st.slider("Cholesterol (mg/dL)", 100.0, 500.0, 250.0)
albumin = st.slider("Albumin (g/dL)", 1.0, 6.0, 3.5)
copper = st.slider("Copper (µg/dL)", 0.0, 300.0, 50.0)
alk_phos = st.slider("Alkalen Fosfataz", 50.0, 3000.0, 500.0)
sgot = st.slider("SGOT (AST)", 10.0, 500.0, 50.0)
trig = st.slider("Trigliserid", 50.0, 500.0, 150.0)
platelets = st.slider("Platelets (10³/µL)", 50.0, 500.0, 250.0)
prothrombin = st.slider("Prothrombin Time", 8.0, 20.0, 12.0)

st.divider()

# =========================
# TAHMİN
# =========================
if st.button("🔍 Siroz Evresini Tahmin Et"):

    sex_val = 1 if sex == "Male" else 0
    status_val = 1 if status == "D" else 0
    drug_val = 1 if drug == "D-penicillamine" else 0

    ascites_val = 1 if ascites == "Var" else 0
    hepatomegaly_val = 1 if hepatomegaly == "Var" else 0
    spiders_val = 1 if spiders == "Var" else 0
    edema_val = int(edema)

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

    input_df = input_df[model.feature_names_in_]

    pred = model.predict(input_df)
    probs = model.predict_proba(input_df)[0]
    stage = le_stage.inverse_transform(pred)[0]

    st.markdown("---")

    st.markdown(
        f"""
        <div style="
            background-color:#ffffff;
            padding:25px;
            border-radius:12px;
            box-shadow:0 0 12px rgba(0,0,0,0.08);
            text-align:center;
        ">
            <h2>🧬 Tahmin Edilen Siroz Evresi</h2>
            <h1 style="color:#1f77b4;">Stage {stage}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📊 Evre Olasılık Dağılımı")
    st.caption("Modelin her evre için hesapladığı olasılıklar")

    for s, p in zip(le_stage.classes_, probs):
        st.progress(float(p), text=f"Stage {s}: %{p*100:.2f}")
