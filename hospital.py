import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Hospital Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- HOSPITAL IMAGE BACKGROUND --------------------
page_bg = """
<style>

/* MAIN BACKGROUND WITH HOSPITAL IMAGE */
[data-testid="stAppViewContainer"] {
    background-image: 
        linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)),
        url("https://images.unsplash.com/photo-1586773860418-d37222d8fce3");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Segoe UI', sans-serif;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b5ed7, #0d6efd);
}
[data-testid="stSidebar"] * {
    color: white;
}

/* HEADINGS */
h1 {
    color: #0b5ed7;
    font-weight: 800;
}
h2, h3 {
    color: #0d6efd;
}

/* CARD STYLE (CONTENT BOX) */
.card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
    border-left: 6px solid #0d6efd;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 14px;
}

/* INPUT BOXES */
.stNumberInput input {
    border-radius: 10px;
    border: 1px solid #90caf9;
}

/* BUTTON */
.stButton>button {
    background: #0d6efd;
    color: white;
    font-size: 16px;
    font-weight: 600;
    border-radius: 10px;
    padding: 10px 22px;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    background: #084298;
    transform: scale(1.05);
}

/* ALERTS */
.stAlert {
    border-radius: 14px;
    font-size: 16px;
}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.title("🏥 Hospital Disease Prediction App")
st.write(
    "A medical prediction system that uses **Machine Learning** "
    "to detect disease based on patient health parameters."
)

# -------------------- LOAD DATA --------------------
uploaded_file = st.file_uploader("📂 Upload Hospital Data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Dataset preview
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    st.markdown("</div>", unsafe_allow_html=True)

    # Grouped stats
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Average Values by Disease")
    st.write(df.groupby("Disease").mean())
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- MODEL TRAINING --------------------
    X = df[["Age", "Fever", "BP", "Sugar"]]
    y = df["Disease"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.success(f"✅ Model Accuracy: {accuracy*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- USER INPUT --------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧑‍⚕️ Enter Patient Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 20)
        fever = st.number_input("Fever (°F)", 90, 110, 98)

    with col2:
        bp = st.number_input("Blood Pressure (mmHg)", 80, 200, 120)
        sugar = st.number_input("Sugar Level (mg/dL)", 50, 300, 100)

    if st.button("🔍 Predict Disease"):
        new_data = pd.DataFrame(
            [[age, fever, bp, sugar]],
            columns=["Age", "Fever", "BP", "Sugar"]
        )

        prediction = model.predict(new_data)
        result = le.inverse_transform(prediction)

        if result[0].lower() in ["yes", "disease", "positive"]:
            st.error("🩺 Disease Detected")
        else:
            st.success("✅ No Disease Detected")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("📂 Please upload a Hospital_data.csv file to proceed.")
