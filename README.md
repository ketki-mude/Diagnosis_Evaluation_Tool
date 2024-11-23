# Diagnosis_Evaluation_Tool
Real-Time Medical Diagnosis Evaluation Tool

---
### Project Description:

- **Symptom-Based Diagnostic System**:
   Leveraging a curated medical dataset, we trained multiple models, including Random Forest, K-Nearest Neighbors (KNN), Logistic Regression (LR), and Support Vector Machine (SVM), to provide accurate predictions based on user symptoms. By deploying these models in an ensemble configuration, we ensure robust and consistent diagnostic suggestions for a wide range of symptom inputs.

- **Deployment and Scaling on AWS**:
   The trained ensemble model, integrated with a language model for symptom-based natural language responses, is deployed on AWS. The deployment is optimized for scalability, handling over 500 concurrent users with minimal latency, providing quick and reliable diagnostic assistance even under high traffic.

- **Ensemble Voting for Enhanced Accuracy**:
   Our system uses an ensemble voting approach to combine predictions from multiple models (Random Forest, KNN, LR, and SVM), selecting the final diagnosis based on the most common or weighted vote among the models. This approach boosts the model's predictive accuracy, improving consistency and reliability in diagnostics.

- **Streamlit Interface for Real-Time Diagnostics**:
   A user-friendly Streamlit-based interface allows users to enter symptoms and receive real-time diagnostic feedback. The interface is designed for intuitive use, providing accessible diagnostic assistance with clear prompts and easy navigation.

- **Automated Pipeline for Continuous Model Improvement**:
   An automated pipeline orchestrates data processing, model retraining, and accuracy monitoring, leading to a consistent increase in predictive accuracy—from 85% to 95%. This improvement significantly boosts user trust and engagement, as the model continues to learn and adapt from new data inputs over time.

### Summary:

This end-to-end solution provides an accessible, scalable, and accurate tool for preliminary medical diagnosis. It combines multiple machine learning algorithms in an ensemble voting system, a responsive Streamlit interface, and a scalable deployment on AWS. The automated pipeline further supports ongoing accuracy improvements, reinforcing the system’s trustworthiness and user engagement over time.
