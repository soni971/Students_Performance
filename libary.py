import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# App title
st.title("📚 Library Frequent User Prediction App")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("library_data.csv")

df = load_data()

# Show dataset
st.subheader("📄 Dataset Preview")
st.dataframe(df)

# Groupby result
st.subheader("📊 Average values by Frequent User")
st.write(df.groupby("FrequentUser").mean())

# Features & Target
X = df[["StudentAge", "BooksIssued", "LateReturns", "MembershipYears"]]
y = df["FrequentUser"]

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Prediction & Accuracy
pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)

st.subheader("🎯 Model Accuracy")
st.write(f"Accuracy of model: **{accuracy * 100:.2f}%**")

# User input section
st.subheader("🧑‍🎓 Predict New Student")

student_age = st.number_input("Student Age", min_value=5, max_value=100, value=10)
books_issued = st.number_input("Books Issued", min_value=0, value=1)
late_returns = st.number_input("Late Returns", min_value=0, value=3)
membership_years = st.number_input("Membership Years", min_value=0, value=4)

# Prediction button
if st.button("Predict"):
    new_data = [[student_age, books_issued, late_returns, membership_years]]
    prediction = model.predict(new_data)

    if prediction[0] == 1:
        st.error("❌ Frequent User : NO")
    else:
        st.success("✅ Frequent User : YES")