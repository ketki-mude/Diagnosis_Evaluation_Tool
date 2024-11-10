import streamlit as st
from openai_integration import extract_symptoms_from_text
from ensemble_voting import ensemble_voting

# Set the page configuration first
st.set_page_config(layout="wide", page_title="Diagnosis Tool")

# Streamlit UI setup
st.sidebar.markdown("<h1 style='text-align: center; color: white;'>Diagnosis Evaluation Tool</h1>", unsafe_allow_html=True)

page = "Home"  # Default page

if st.sidebar.button("🏠 Home", use_container_width=True):
    page = "Home"
if st.sidebar.button("📜 History", use_container_width=True):
    page = "History"

# Main Window Logic Based on Page Selection
if page == "Home":
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Welcome to the Medical Diagnosis Tool</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #4CAF50;'>Type your symptoms below to get a diagnosis.</p>", unsafe_allow_html=True)

    # Design an input form for user symptoms
    symptoms_input = st.text_input(
        "Describe your symptoms in English (max 5 symptoms):",
        placeholder="e.g., persistent cough, fever, headache",
        help="Please provide a clear description of your symptoms."
    )

    col1, col2, col3 = st.columns([3, 1, 3])  # Create three columns
    diagnosis_result = None  # Initialize result variable to store diagnosis information

    # Place button in the center column (col2)
    with col2:
        if st.button("Get Diagnosis", use_container_width=True):
            # Process the symptoms input and store the result
            if symptoms_input:
                diagnosis_result = extract_symptoms_from_text(symptoms_input)

    # Display the response outside the button column, in a new section
    if diagnosis_result:
        # Check if there's an error in the diagnosis result
        if "error" in diagnosis_result:
            st.markdown(
                f'<p style="color:red; font-size:16px; text-align:left;">{diagnosis_result["error"]}</p>',
                unsafe_allow_html=True
            )
        else:
            symptoms = diagnosis_result["symptoms"]
            input_vector = diagnosis_result["input_vector"]
            best_model, prediction = ensemble_voting(input_vector)

            # Display the diagnosis result as plain text
            st.write(f"Based on the symptoms, you might have: {prediction}")
            st.write(f"The diagnosis was determined using the {best_model} model with the highest accuracy.")

elif page == "History":
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Diagnosis History</h2>", unsafe_allow_html=True)
    st.write("This section will display a history of previous diagnoses (feature under construction).")
