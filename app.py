"""Console program for predicting a student's pass/fail result."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"

def train_model():
    """Load the data, train the classifier, and return model accuracy."""
    data = pd.read_csv(DATA_PATH)
    features = ["study_hours", "attendance", "previous_score", "assignments_completed"]
    x = data[features]
    y = data["result"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    return model, accuracy


MODEL, MODEL_ACCURACY = train_model()


def validate_values(study_hours, attendance, previous_score, assignments_completed):
    """Validate input ranges before a prediction is made."""
    if not 0 <= study_hours <= 24:
        raise ValueError("Study hours must be between 0 and 24.")
    if not 0 <= attendance <= 100 or not 0 <= previous_score <= 100:
        raise ValueError("Attendance and previous score must be between 0 and 100.")
    if not 0 <= assignments_completed <= 20:
        raise ValueError("Completed assignments must be between 0 and 20.")


def get_input_values():
    """Ask the user for the four features needed by the model."""
    print("Enter the student's details:")
    study_hours = float(input("Daily study hours: "))
    attendance = float(input("Attendance percentage: "))
    previous_score = float(input("Previous score percentage: "))
    assignments_completed = int(input("Assignments completed: "))
    validate_values(study_hours, attendance, previous_score, assignments_completed)
    return study_hours, attendance, previous_score, assignments_completed


def main():
    """Train the model, collect input in the terminal, and display a result."""
    parser = argparse.ArgumentParser(description="Predict student pass or fail result.")
    parser.add_argument("--study-hours", type=float)
    parser.add_argument("--attendance", type=float)
    parser.add_argument("--previous-score", type=float)
    parser.add_argument("--assignments-completed", type=int)
    args = parser.parse_args()
    provided_values = [args.study_hours, args.attendance, args.previous_score, args.assignments_completed]

    try:
        if any(value is not None for value in provided_values):
            if any(value is None for value in provided_values):
                parser.error("Provide all four command-line options, or provide none for interactive mode.")
            values = tuple(provided_values)
            validate_values(*values)
        else:
            values = get_input_values()

        sample = pd.DataFrame(
            [values],
            columns=["study_hours", "attendance", "previous_score", "assignments_completed"],
        )
        result = MODEL.predict(sample)[0]
        confidence = max(MODEL.predict_proba(sample)[0]) * 100
        print(f"\nPrediction: {result}")
        print(f"Model confidence: {confidence:.1f}%")
        print(f"Validation accuracy on the included demo data: {MODEL_ACCURACY:.1%}")
    except ValueError as error:
        print(f"Input error: {error}")


if __name__ == "__main__":
    main()
