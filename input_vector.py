def find_index(symptoms_array, all_symptoms):
    input_vector = []
    for i in range(0, len(all_symptoms)):
        input_vector.append(0)
        
    for i in range(0, len(symptoms_array)):
        for j in range(0, len(all_symptoms)):
            if symptoms_array[i] == all_symptoms[j]:
                input_vector[j] = 1
                
    return input_vector

 