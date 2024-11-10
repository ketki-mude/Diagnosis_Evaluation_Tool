import joblib
import numpy as np

# Load models and accuracies
models = {
    'random_forest': joblib.load('random_forest.joblib'),
    'logistic_regression': joblib.load('logistic_regression.joblib'),
    'svm': joblib.load('svm.joblib'),
    'knn': joblib.load('knn.joblib')
}

# Load model accuracies
model_accuracies = joblib.load('model_accuracies.joblib')

def ensemble_voting(input_vector):
    best_model_name = None
    best_model_accuracy = 0
    best_prediction = None

    # Iterate over models and find the one with the highest accuracy
    for model_name, model in models.items():
        prediction = model.predict([input_vector])
        accuracy = model_accuracies[model_name]
        print(f"Model: {model_name}, Prediction: {prediction[0]}, Accuracy: {accuracy * 100:.2f}%")
        
        # Update the best model if this model has a higher accuracy
        if accuracy > best_model_accuracy:
            best_model_accuracy = accuracy
            best_model_name = model_name
            best_prediction = prediction[0]

    print(f"Best Model: {best_model_name}, Accuracy: {best_model_accuracy * 100:.2f}%")
    return best_model_name, best_prediction
