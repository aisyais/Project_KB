
from data_loader import load_data
from preprocessing import preprocess_data
from model import split_data, train_and_evaluate_model

def main():
    data_path = r"D:\college\sem 3\kecerdasan buatan\project_uas\Project_Numeric\country-wise-average.csv"

  
    data = load_data(data_path)
 
    X, y = preprocess_data(data)  

    
    X_train, X_test, y_train, y_test = split_data(X, y)

    train_and_evaluate_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
