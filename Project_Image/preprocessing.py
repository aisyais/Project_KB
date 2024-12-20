from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Flatten image data
def flatten_data(train_data, test_data):
    train_data_flat = train_data.reshape(train_data.shape[0], -1)
    test_data_flat = test_data.reshape(test_data.shape[0], -1)
    return train_data_flat, test_data_flat

# Scale features
def scale_data(train_data_flat, test_data_flat):
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data_flat)
    test_data_scaled = scaler.transform(test_data_flat)
    return train_data_scaled, test_data_scaled

# Handle class imbalance
def balance_data(train_data_flat, train_labels):
    ros = RandomOverSampler(random_state=42)
    train_data_balanced, train_labels_balanced = ros.fit_resample(train_data_flat, train_labels)
    return train_data_balanced, train_labels_balanced
