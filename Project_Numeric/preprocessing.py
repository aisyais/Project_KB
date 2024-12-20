import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    """Preprocess the dataset by handling missing values and encoding."""
    X = data.drop(columns=['Country', 'Income Classification'])
    y = data['Income Classification']

    # Imputer untuk menangani missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Encode kolom 'Country'
    label_encoder = LabelEncoder()
    data['Country'] = label_encoder.fit_transform(data['Country'])

    return X, y  # Hanya mengembalikan X dan y
