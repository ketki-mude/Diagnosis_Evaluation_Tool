# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset
df = pd.read_csv('diagnosis_dataset_v2.csv')

# Step 2: Split the dataset into features (X) and target (y)
X = df.drop('prognosis', axis=1)  # Replace 'target_column' with the name of your target column
y = df['prognosis']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train individual models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000)
svm = SVC(probability=True)  # Enable probability=True for soft voting
knn = KNeighborsClassifier()

# Step 5: Create the Voting Classifier (Hard or Soft Voting)
# Hard Voting
voting_clf_hard = VotingClassifier(estimators=[
    ('rf', rf), ('lr', lr), ('svm', svm), ('knn', knn)], voting='hard')

# Soft Voting (Use this if you want to average predicted probabilities)
voting_clf_soft = VotingClassifier(estimators=[
    ('rf', rf), ('lr', lr), ('svm', svm), ('knn', knn)], voting='soft')

# Step 6: Train the Voting Classifier
voting_clf_hard.fit(X_train, y_train)
voting_clf_soft.fit(X_train, y_train)

# Step 7: Evaluate each model and the ensemble
# Hard Voting
y_pred_hard = voting_clf_hard.predict(X_test)
print("Hard Voting Classifier Accuracy:", accuracy_score(y_test, y_pred_hard))
print("Classification Report for Hard Voting:\n", classification_report(y_test, y_pred_hard))

# Soft Voting
y_pred_soft = voting_clf_soft.predict(X_test)
print("\nSoft Voting Classifier Accuracy:", accuracy_score(y_test, y_pred_soft))
print("Classification Report for Soft Voting:\n", classification_report(y_test, y_pred_soft))

# Optional: Evaluate individual models
for clf, name in zip([rf, lr, svm, knn], ['Random Forest', 'Logistic Regression', 'SVM', 'KNN']):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))
