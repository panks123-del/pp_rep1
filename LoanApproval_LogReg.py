################ USE CASE #######################
# Loan Approval application
# Given the input data about financial condition of a loan applicant, should you approve the loan.
# Find the probability that the applicant will payback the loan. 
# Then calculate the appropriate amount of loan that should be given to the applicant to have high chance of payback

################ STEPS ##########################
# Import the necessary libraries
# Read the datafile to build the logic
# Check the null values in the datafile. If there are null values, remove them.
# Drop the columns which will not be used in the model
# Make sure that the data is stripped off the white spaces
# Change the categorical variable to numeric ones
# Define x and y
# Do the scaling for numerical columns
# Split the data into training and test
# build the model using training data
# Run the model on test data
# Calculate accuracy
# Calculate confusion matrix parameters and plot
# Create a classification report
# Test for a new loan aplication by taking raw inputs as dictionary
# Change the dictionary into datafram
# Select only the numerical columns for scaling and apply the same scaler
# Make a prediction with the trained model
# Then suggest the amount of loan = probability * loan applied for

import numpy as np
import pandas as pd
import os
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df=pd.read_csv('loan_approval_dataset.csv')
#df.head()
df.isnull().sum()
#df.dtypes
df.drop(columns=['loan_id'], inplace=True)
#print(df.columns)
df.rename(columns=lambda x: x.strip(), inplace=True)
#print(df.columns)

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Apply label encoding to the 'education' column
df['education'] = label_encoder.fit_transform(df['education'])

# Apply label encoding to the 'self_employed' column
df['self_employed'] = label_encoder.fit_transform(df['self_employed'])

# Apply label encoding to the 'loan_status' column
df['loan_status'] = label_encoder.fit_transform(df['loan_status'])

# Display the updated DataFrame with encoded columns
#print(df[['education', 'self_employed','loan_status']])

#df.dtypes

# Create a StandardScaler instance
scaler = StandardScaler()

# Define the feature columns (X) and target column (y)
x = df.drop(columns=['loan_status'])  # Drop 'loan_status' column to get feature columns
y = df['loan_status']  # Target variable

# Select only the numerical columns for scaling (excluding 'loan_status')
numerical_columns = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                      'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value',
                      'bank_asset_value']

# Apply scaling to the numerical columns
x[numerical_columns] = scaler.fit_transform(x[numerical_columns])

# Display the scaled feature variables (X) and the target variable (y)
#print("Scaled Feature Variables (x):")
#print(x.head())

#print("\nTarget Variable (y):")
#print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape)
print(x_test.shape)


# Create a LogisticRegression instance
logistic_reg = LogisticRegression(random_state=42)

# Train the logistic regression model
logistic_reg.fit(x_train, y_train)

# Predict on the test set
y_pred = logistic_reg.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Generate a classification report
classification_rep = classification_report(y_test, y_pred)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Print evaluation metrics separately
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Print the classification report
print("Classification Report:\n", classification_rep)


# Add the predicted loan approval probabilities (class 1) to x_test
x_test_with_proba = x_test.copy()
y_pred_proba = logistic_reg.predict_proba(x_test)
#print("Predicted Probabilities for each class (0 and 1):\n", y_pred_proba[:5])
#print("\nPredicted Probabilities for class 1 (loan approval):\n", y_pred_proba[:5, 1])

x_test_with_proba['loan_approval_probability'] = y_pred_proba[:, 1]

# Get the unscaled loan_amount from the original DataFrame using the test set indices
x_test_with_proba['original_loan_amount'] = df.loc[x_test.index, 'loan_amount']

# Calculate the 'average loan amount' by multiplying probability with the original loan_amount
x_test_with_proba['avg_loan_amount_pred'] = x_test_with_proba['loan_approval_probability'] * x_test_with_proba['original_loan_amount']

print("x_test DataFrame with new 'loan_approval_probability', 'original_loan_amount', and 'avg_loan_amount_pred' columns:")
display(x_test_with_proba.head())


# Example new loan application data as a dictionary
new_loan_data = {
    'no_of_dependents': 2,
    'education': 0,  # 0 for Graduate, 1 for Not Graduate (based on LabelEncoder for education)
    'self_employed': 0, # 0 for No, 1 for Yes (based on LabelEncoder for self_employed)
    'income_annum': 7000000,
    'loan_amount': 20000000,
    'loan_term': 12,
    'cibil_score': 750,
    'residential_assets_value': 8000000,
    'commercial_assets_value': 2000000,
    'luxury_assets_value': 10000000,
    'bank_asset_value': 5000000
}

# Convert the dictionary to a pandas DataFrame
new_loan_df = pd.DataFrame([new_loan_data])

print("New Loan Application DataFrame:")
display(new_loan_df)

# Select only the numerical columns for scaling in the new loan data
new_loan_numerical_data = new_loan_df[numerical_columns]

# Apply the same scaler used for the training data to the new loan's numerical data
scaled_new_loan_numerical_data = scaler.transform(new_loan_numerical_data)

# Create a DataFrame from the scaled numerical data to maintain column names
scaled_new_loan_df = pd.DataFrame(scaled_new_loan_numerical_data, columns=numerical_columns)

# Combine scaled numerical columns with original categorical columns (education, self_employed)
# Ensure the order of columns matches the training data (x)

# First, copy the entire new_loan_df to retain all columns (including non-numerical for later, if needed)
final_new_loan_df = new_loan_df.copy()

# Replace the numerical columns with their scaled versions
for col in numerical_columns:
    final_new_loan_df[col] = scaled_new_loan_df[col]

# Reorder columns to match the 'x' DataFrame that the model was trained on
final_new_loan_df = final_new_loan_df[x.columns]

# Make a prediction with the trained model
prediction = logistic_reg.predict(final_new_loan_df)
prediction_proba = logistic_reg.predict_proba(final_new_loan_df)

print("Scaled New Loan Application Data:")
display(final_new_loan_df)

print(f"\nPredicted Loan Status: {prediction[0]} (0 = Rejected, 1 = Approved)")
print(f"Probability of Loan Approval: {prediction_proba[0][1]:.4f}")

# Calculate the recommended loan amount
recommended_loan_amount = prediction_proba[0][1] * new_loan_df['loan_amount'].iloc[0]
print(f"Recommended Loan Amount (Probability * Applied Amount): {recommended_loan_amount:,.2f}")
