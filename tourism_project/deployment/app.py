"""
Streamlit front-end for the Wellness Tourism Package purchase predictor.

The app loads the model that the GitHub Actions pipeline trained and committed
into this folder, collects the customer's details through form widgets, builds a
single-row DataFrame with exactly the columns used in training, and shows the
predicted purchase likelihood.
"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "best_tourism_model_v1.joblib")
METRICS_PATH = os.path.join(HERE, "model_metrics.json")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")
st.title("Wellness Tourism Package Purchase Prediction")
st.write(
    """
    **Visit with Us** uses this tool to decide which customers to contact about the new
    *Wellness Tourism Package*. Enter a customer's profile and past interaction details
    below, and the model will predict whether they are likely to purchase the package.
    """
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model not found. Run the GitHub Actions pipeline first so it commits "
        "`best_tourism_model_v1.joblib` into `tourism_project/deployment/`."
    )
    st.stop()

model = load_model()

# ---- Customer details ----------------------------------------------------------
st.header("Customer Details")
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    citytier = st.selectbox("City Tier", [1, 2, 3], index=0)
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    maritalstatus = st.selectbox(
        "Marital Status", ["Married", "Single", "Unmarried", "Divorced"]
    )
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
with c2:
    monthlyincome = st.number_input(
        "Monthly Income", min_value=1000.0, max_value=200000.0, value=22000.0, step=500.0
    )
    numberofpersonvisiting = st.number_input(
        "Number of Persons Visiting", min_value=1, max_value=10, value=3, step=1
    )
    numberofchildrenvisiting = st.number_input(
        "Number of Children Visiting (below 5)", min_value=0, max_value=5, value=1, step=1
    )
    preferredpropertystar = st.selectbox("Preferred Property Star", [3, 4, 5])
    numberoftrips = st.number_input(
        "Number of Trips per Year", min_value=0, max_value=30, value=3, step=1
    )
    passport = st.selectbox("Has Passport", ["Yes", "No"])
    owncar = st.selectbox("Owns a Car", ["Yes", "No"])

# ---- Interaction details -------------------------------------------------------
st.header("Sales Interaction Details")
c3, c4 = st.columns(2)
with c3:
    productpitched = st.selectbox(
        "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
with c4:
    numberoffollowups = st.number_input(
        "Number of Follow-ups", min_value=0, max_value=10, value=3, step=1
    )
    durationofpitch = st.number_input(
        "Duration of Pitch (minutes)", min_value=1, max_value=180, value=15, step=1
    )

# ---- Assemble the inputs into a DataFrame with the training column names -------
input_data = pd.DataFrame([{
    "Age": float(age),
    "TypeofContact": typeofcontact,
    "CityTier": int(citytier),
    "DurationOfPitch": float(durationofpitch),
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": int(numberofpersonvisiting),
    "NumberOfFollowups": float(numberoffollowups),
    "ProductPitched": productpitched,
    "PreferredPropertyStar": float(preferredpropertystar),
    "MaritalStatus": maritalstatus,
    "NumberOfTrips": float(numberoftrips),
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": int(pitchsatisfactionscore),
    "OwnCar": 1 if owncar == "Yes" else 0,
    "NumberOfChildrenVisiting": float(numberofchildrenvisiting),
    "Designation": designation,
    "MonthlyIncome": float(monthlyincome),
}])

with st.expander("Show model input"):
    st.dataframe(input_data)

# ---- Predict -------------------------------------------------------------------
if st.button("Predict Purchase", type="primary"):
    proba = float(model.predict_proba(input_data)[0, 1])
    prediction = int(proba >= 0.5)

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"This customer is **likely to purchase** the Wellness Tourism Package. "
                   f"(Purchase probability: {proba:.1%})")
    else:
        st.info(f"This customer is **unlikely to purchase** the Wellness Tourism Package. "
                f"(Purchase probability: {proba:.1%})")
    st.progress(proba)

# ---- Model info ----------------------------------------------------------------
if os.path.exists(METRICS_PATH):
    with st.expander("Model performance (from the latest pipeline run)"):
        with open(METRICS_PATH) as f:
            info = json.load(f)
        m = info.get("metrics", {})
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Test Accuracy", f"{m.get('test_accuracy', 0):.3f}")
        k2.metric("Test Precision", f"{m.get('test_precision', 0):.3f}")
        k3.metric("Test Recall", f"{m.get('test_recall', 0):.3f}")
        k4.metric("Test F1", f"{m.get('test_f1', 0):.3f}")
        st.json(info.get("best_params", {}))
