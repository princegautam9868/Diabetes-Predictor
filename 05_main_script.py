import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------
# IMPORTANT:
# Keep diabetes.csv in the same folder as this script.
# Do NOT use a Windows path such as E:\Diabetes_prediction\...
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "diabetes.csv was not found. "
        "Please upload/place diabetes.csv in the same folder as "
        "05_main_script.py."
    )
    st.stop()


# ---------------------------------------------------------
# PAGE HEADINGS
# ---------------------------------------------------------
st.title("🩺 Diabetes Checkup")
st.sidebar.header("Patient Data")

st.subheader("Training Data Statistics")
st.write(df.describe())


# ---------------------------------------------------------
# X AND Y DATA
# ---------------------------------------------------------
x = df.drop(["Outcome"], axis=1)
y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=0,
    stratify=y
)


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------
def user_report():
    pregnancies = st.sidebar.slider(
        "Pregnancies", 0, 17, 3
    )

    glucose = st.sidebar.slider(
        "Glucose", 0, 200, 120
    )

    blood_pressure = st.sidebar.slider(
        "Blood Pressure", 0, 122, 70
    )

    skin_thickness = st.sidebar.slider(
        "Skin Thickness", 0, 100, 20
    )

    insulin = st.sidebar.slider(
        "Insulin", 0, 846, 79
    )

    bmi = st.sidebar.slider(
        "BMI", 0.0, 67.0, 20.0
    )

    diabetes_pedigree_function = st.sidebar.slider(
        "Diabetes Pedigree Function", 0.0, 2.4, 0.47
    )

    age = st.sidebar.slider(
        "Age", 21, 88, 33
    )

    user_report_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age
    }

    return pd.DataFrame(user_report_data, index=[0])


# ---------------------------------------------------------
# PATIENT DATA
# ---------------------------------------------------------
user_data = user_report()

st.subheader("Patient Data")
st.write(user_data)


# ---------------------------------------------------------
# MACHINE LEARNING MODEL
# ---------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(x_train, y_train)

user_result = rf.predict(user_data)

model_accuracy = accuracy_score(
    y_test,
    rf.predict(x_test)
)


# ---------------------------------------------------------
# VISUALISATIONS
# ---------------------------------------------------------
st.title("Visualised Patient Report")

color = "blue" if user_result[0] == 0 else "red"


def display_graph(
    title,
    y_column,
    palette,
    y_ticks=None,
    y_limit=None
):
    fig, ax = plt.subplots()

    sns.scatterplot(
        x="Age",
        y=y_column,
        data=df,
        hue="Outcome",
        palette=palette,
        ax=ax
    )

    ax.scatter(
        user_data["Age"],
        user_data[y_column],
        s=150,
        color=color,
        label="Your Data"
    )

    ax.set_title("0 - Healthy & 1 - Diabetic")
    ax.set_xlabel("Age")

    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    if y_limit is not None:
        ax.set_ylim(y_limit)

    ax.legend()

    st.pyplot(fig)
    plt.close(fig)


# Age vs Pregnancies
st.header("Pregnancy Count Graph (Others vs Yours)")
display_graph(
    "Pregnancy Count",
    "Pregnancies",
    "Greens",
    np.arange(0, 20, 2),
    (0, 20)
)


# Age vs Glucose
st.header("Glucose Value Graph (Others vs Yours)")
display_graph(
    "Glucose",
    "Glucose",
    "magma",
    np.arange(0, 221, 20),
    (0, 220)
)


# Age vs Blood Pressure
st.header("Blood Pressure Value Graph (Others vs Yours)")
display_graph(
    "Blood Pressure",
    "BloodPressure",
    "Reds",
    np.arange(0, 131, 10),
    (0, 130)
)


# Age vs Skin Thickness
st.header("Skin Thickness Value Graph (Others vs Yours)")
display_graph(
    "Skin Thickness",
    "SkinThickness",
    "Blues",
    np.arange(0, 111, 10),
    (0, 110)
)


# Age vs Insulin
st.header("Insulin Value Graph (Others vs Yours)")
display_graph(
    "Insulin",
    "Insulin",
    "rocket",
    np.arange(0, 901, 100),
    (0, 900)
)


# Age vs BMI
st.header("BMI Value Graph (Others vs Yours)")
display_graph(
    "BMI",
    "BMI",
    "rainbow",
    np.arange(0, 71, 5),
    (0, 70)
)


# Age vs Diabetes Pedigree Function
st.header("DPF Value Graph (Others vs Yours)")
display_graph(
    "Diabetes Pedigree Function",
    "DiabetesPedigreeFunction",
    "YlOrBr",
    np.arange(0, 3.01, 0.2),
    (0, 3)
)


# ---------------------------------------------------------
# FINAL OUTPUT
# ---------------------------------------------------------
st.subheader("Your Report")

if user_result[0] == 0:
    output = "You are not Diabetic"
else:
    output = "You are Diabetic"

st.title(output)

st.subheader("Model Accuracy")
st.write(f"{model_accuracy * 100:.2f}%")

st.info(
    "This application is for educational purposes only and "
    "is not a medical diagnosis."
)
