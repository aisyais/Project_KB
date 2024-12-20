
import data_loader
import preprocessing
import model

if __name__ == "__main__":
    # Path to your dataset folder
    dataset_path = r"D:\college\sem 3\kecerdasan buatan\project_uas\Project_Image\datasetsignature"

    # Load data
    (train_data, test_data, train_labels, test_labels), class_names = data_loader.load_data(dataset_path)

    # Validate dataset
    if len(set(train_labels)) != len(class_names) or len(set(test_labels)) != len(class_names):
        print("Warning: Some classes are missing in the training or test set!")

    # Preprocess data
    train_data_flat, test_data_flat = preprocessing.flatten_data(train_data, test_data)
    train_data_scaled, test_data_scaled = preprocessing.scale_data(train_data_flat, test_data_flat)

    # Handle class imbalance
    train_data_balanced, train_labels_balanced = preprocessing.balance_data(train_data_scaled, train_labels)

    # Train model
    svm_model = model.train_svm(train_data_balanced, train_labels_balanced)

    # Evaluate model
    accuracy, report = model.evaluate_model(svm_model, test_data_scaled, test_labels, class_names)
    print(f"Accuracy: {accuracy:.2f}%")
    print("Classification Report:")
    print(report)
