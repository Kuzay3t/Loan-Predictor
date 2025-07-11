# Loan-Predictor

🏦 Loan Approval Predictor – Supervised Machine Learning
This project is a Loan Approval Predictor that uses Supervised Learning techniques to determine whether a person is eligible for a loan based on their financial and personal features. Built and tested in Google Colab, this notebook walks you through data preprocessing, exploratory data analysis, model building, and evaluation.

📌 Objective
To predict loan eligibility using a classification model trained on a dataset of loan applicants, with features like income, credit history, and employment status.

📁 Dataset
A miniature version of the dataset is provided for quick testing and demonstration purposes. The dataset typically includes the following columns:

Feature Name	Description
Gender	Male/Female
Married	Marital status
ApplicantIncome	Income of the loan applicant
CoapplicantIncome	Income of co-applicant
LoanAmount	Loan amount requested
Loan_Amount_Term	Term of the loan (in months)
Credit_History	1 if the person has a good credit history
Property_Area	Urban/Semiurban/Rural
Education	Graduate/Not Graduate
Self_Employed	Yes/No
Loan_Status	Target variable: Y (Yes) / N (No)

🔧 Installation & Setup
You can run this project directly in Google Colab.

Open the .ipynb notebook in Colab.

Upload the dataset (CSV file).

Run each code cell sequentially.

No special installations are required beyond the usual data science libraries like pandas, numpy, sklearn, and matplotlib.

🔍 Code Walkthrough & Explanation
📥 1. Importing Libraries
python
Copy
Edit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
This block imports all the essential libraries for data handling, visualization, and machine learning.

📄 2. Loading the Dataset
python
Copy
Edit
df = pd.read_csv("loan_data.csv")
df.head()
Loads the dataset using pandas. You can inspect the top rows using head() to understand the structure.

🧼 3. Data Preprocessing
python
Copy
Edit
df.isnull().sum()
df.fillna(method='ffill', inplace=True)
Checks for missing values.

Fills missing entries using forward fill (propagates last valid value).

Next:

python
Copy
Edit
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    df[col] = le.fit_transform(df[col])
Encodes categorical features into numeric values using LabelEncoder, which is required for machine learning algorithms.

📊 4. Exploratory Data Analysis (EDA)
python
Copy
Edit
sns.countplot(x='Loan_Status', data=df)
plt.title('Loan Approval Distribution')
This visualizes the target variable – how many applicants were approved vs denied.

You can include more plots:

sns.boxplot() for outlier analysis

sns.heatmap() for correlation between variables

🧠 5. Feature Selection & Splitting
python
Copy
Edit
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Drops the target variable from the features.

Splits the data into training and testing sets (80% / 20%).

📈 6. Model Training – Logistic Regression
python
Copy
Edit
model = LogisticRegression()
model.fit(X_train, y_train)
This builds a Logistic Regression classifier. You can experiment with other models like DecisionTreeClassifier, RandomForestClassifier, etc.

🧪 7. Model Evaluation
python
Copy
Edit
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
Evaluates model performance:

Accuracy: Overall correctness

Classification Report: Precision, Recall, F1-Score

Confusion Matrix: Visual insight into prediction errors

✅ 8. Conclusion
Your model is now capable of predicting whether an applicant is likely to be approved for a loan based on their financial history and demographic profile.

🚀 Future Improvements
Use hyperparameter tuning (e.g., GridSearchCV)

Try other models: Random Forest, XGBoost

Improve data imputation for missing values

Use original (full) dataset for better generalization

📌 Folder Structure (Example)
Copy
Edit
loan-approval-predictor/
├── loan_data.csv
├── Loan_Predictor.ipynb
├── README.md
└── output_images/
🤝 Contributing
Feel free to fork this repository, use your own dataset, try different models, and submit pull requests!

🧠 Author
Kuzayet – [Your LinkedIn or GitHub link here]

