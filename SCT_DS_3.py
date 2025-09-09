import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

with zipfile.ZipFile("/content/bank.zip", "r") as zip_ref:
    zip_ref.extractall("/content/")

df = pd.read_csv("/content/bank-full.csv", sep=';')

label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Classifier to Predict Customer Purchase\n")
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(
    y_test, y_pred, target_names=['No', 'Yes'], output_dict=True
)
print("\nPurchase Prediction Metrics (Class: 'Yes')")
print(f"Precision: {report['Yes']['precision']:.2f}")
print(f"Recall: {report['Yes']['recall']:.2f}")
print(f"F1-score: {report['Yes']['f1-score']:.2f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title("Confusion Matrix - Customer Purchase Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree for Customer Purchase Prediction")
plt.tight_layout()
plt.show()
