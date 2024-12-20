from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Train the Random Forest model and evaluate it."""
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    # Mendapatkan kelas unik dari y_train untuk target_names
    target_names = [str(cls) for cls in sorted(set(y_train))]

    report = classification_report(y_test, y_pred, target_names=target_names)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy*100:.2f}%")  # Menampilkan akurasi dalam persen
    print(f"F1 Score: {f1:.2f}")
    print("Classification Report:")
    print(report)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
