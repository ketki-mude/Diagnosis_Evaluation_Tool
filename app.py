import streamlit as st
import joblib
import numpy as np

# Load pre-trained models (ensure these models are saved beforehand)
voting_clf_hard = joblib.load('voting_classifier_hard.pkl')
voting_clf_soft = joblib.load('voting_classifier_soft.pkl')

# Set the overall page layout and style
st.set_page_config(layout="wide", page_title="Diagnosis Tool")

# Sidebar with Navigation
st.sidebar.markdown("<h1 style='text-align: center; color: white;'>Diagnosis Evaluation Tool</h1>", unsafe_allow_html=True)

# Adding buttons for navigation to the sidebar
if st.sidebar.button("🏠 Home", use_container_width=True):
    page = "Home"
if st.sidebar.button("📜 History", use_container_width=True):
    page = "History"

# Main Window Logic Based on Page Selection
if 'page' not in locals():
    page = "Home"

if page == "Home":
    # Main content section
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Welcome to the Medical Diagnosis Tool</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #4CAF50;'>Type your symptoms below to get a diagnosis.</p>", unsafe_allow_html=True)

    # Design an input form for user symptoms
    symptoms = st.text_input(
        "Describe your symptoms in English:",
        placeholder="e.g., persistent cough, fever, headache",
        help="Please provide a clear description of your symptoms."
    )

    # Center-align the "Get Diagnosis" button
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        if st.button("Get Diagnosis", use_container_width=True):
            if symptoms:
                # Simulate feature extraction (you'd have a pipeline to convert symptoms to numerical features)
                input_features = np.array([[0.5, 0.6, 0.7]])  # Dummy features for the example

                # Hard Voting Prediction
                prediction_hard = voting_clf_hard.predict(input_features)[0]

                # Soft Voting Prediction
                prediction_soft = voting_clf_soft.predict(input_features)[0]

                # Display the diagnosis results
                st.success(f"**Hard Voting Diagnosis**: {prediction_hard}")
                st.info(f"**Soft Voting Diagnosis**: {prediction_soft}")

                # Feedback Section
                feedback = st.radio("Was this diagnosis correct?", ["Yes", "No"])
                if feedback:
                    if feedback == "Yes":
                        st.write("Thank you for your feedback! 😊")
                    else:
                        st.write("We appreciate your feedback and will work to improve the tool. 🙏")
            else:
                st.warning("Please provide symptoms to get a diagnosis.")

elif page == "History":
    # Main content for history
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Diagnosis History</h2>", unsafe_allow_html=True)
    st.write("This section will display a history of previous diagnoses (feature under construction).")
