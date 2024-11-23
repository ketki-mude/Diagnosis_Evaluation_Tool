import openai
import pandas as pd
from input_vector import find_index
import os

# OpenAI API Key (use your actual API key)
openai.api_key = 'xxxxxx'

# Load symptom keywords from dataset
symptom_keywords = pd.read_csv('symptom_keywords.csv')['keywords'].tolist()

def extract_symptoms_from_text(text):
    print('Printing text from OpenAI integration extract function: ', text)
    """Use OpenAI to extract relevant symptoms from text input and match with known symptoms."""
    
    symptom_keywords_text = ", ".join(symptom_keywords)
    
    prompt = f"""
    Given this list of valid symptoms: {symptom_keywords_text}
    And this user description: {text}
    Return only the matching symptoms from the list, one per line, without any additional text or punctuation.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a symptom extraction assistant. Only return matching symptoms, one per line, without any additional text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0
    )
    
    # Extract and clean the symptoms
    extracted_text = response['choices'][0]['message']['content'].strip()
    symptoms_array = [symptom.strip() for symptom in extracted_text.split('\n') if symptom.strip()]
    
    all_symptoms_array = []
    if os.path.isfile('symptom_keywords.csv'):
        with open('symptom_keywords.csv', 'r') as file:
            for line in file:
                line = line.replace("\n", '')
                all_symptoms_array.append(line)
    all_symptoms_array = all_symptoms_array[1:]  # Remove the header

    print("Printing symptoms from OpenAI")
    print(symptoms_array)
    print('Printing all symptoms')
    print(all_symptoms_array)
    
    # Check if the symptoms extracted are more than 5
    if len(symptoms_array) > 5:
        return {"error": "❗ Too many symptoms! Please limit your symptoms to 5. If you are experiencing more, please call an ambulance and visit the hospital immediately!"}
    
    # Find the index of symptoms in the symptom keywords
    input_vector = find_index(symptoms_array, all_symptoms_array)
    print("Printing input vector: ", input_vector)
    
    return {"symptoms": symptoms_array, "input_vector": input_vector}
