from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_svm(train_data_scaled, train_labels):
    model = SVC(kernel='linear', C=1, random_state=42)
    model.fit(train_data_scaled, train_labels)
    return model

def evaluate_model(model, test_data_scaled, test_labels, class_names):
    predictions = model.predict(test_data_scaled)
    accuracy = accuracy_score(test_labels, predictions) * 100

    # Ensure all classes are accounted for
    report = classification_report(
        test_labels,
        predictions,
        labels=range(len(class_names)),
        target_names=class_names
    )

    cm = confusion_matrix(test_labels, predictions, labels=range(len(class_names)))

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    return accuracy, report