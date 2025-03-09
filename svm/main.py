from svm import SVM_C
from evaluation import evaluate_model
from dataset_load import load_dataset
from sklearn.model_selection import train_test_split


def main():

    X, y = load_dataset()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Create an SVM classifier
    clf = SVM_C()

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = evaluate_model(y_test, y_pred)['Accuracy']
    print(f'Accuracy: {accuracy:.2f}')

    clf.visualize(X_test, y_test, y_pred)


if __name__ == "__main__":
    main()
