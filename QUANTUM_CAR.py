#%% 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

pd.set_option("future.no_silent_downcasting", True)

# Start timer
start_time = time.time()

# Load the Car Evaluation Dataset
column_names = ["Buying", "Maint", "Doors", "Persons", "Lug_Boot", "Safety", "Class"]

data = pd.read_csv("C:/Users/lucas/OneDrive/Escritorio/Universidad/MASTER/1_SEMESTER/ML/data/car.data", names=column_names)

# Convert categorical values to numerical values
categorical_columns = ["Buying", "Maint", "Doors", "Persons", "Lug_Boot", "Safety", "Class"]
for col in categorical_columns:
    data[col] = data[col].astype("category").cat.codes

# Separate features and target
X = data.drop(columns=["Class"]).values
Y = data["Class"].values

# Split data into train, test, and holdout sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
X_test, X_holdout, Y_test, Y_holdout = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)

print("Train shape =", X_train.shape)
print("Test shape =", X_test.shape)
print("Holdout shape =", X_holdout.shape)

# Train an SVM classifier
selected_kernel = "sigmoid"
clf_svm = SVC(kernel=selected_kernel, C=1.0, gamma="scale")
clf_svm.fit(X_train, Y_train)

# Evaluate performance
accuracy = clf_svm.score(X_test, Y_test)
print(f"Accuracy: {accuracy}")


# End timer
end_time = time.time()
execution_time = end_time - start_time
print(f"Total Execution Time: {execution_time:.2f} seconds")


Y_pred = clf_svm.predict(X_test)
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(conf_matrix)
