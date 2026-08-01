"""Flask app for predicting a student's pass/fail result."""

from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"

app = Flask(__name__)


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


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    values = {"study_hours": "", "attendance": "", "previous_score": "", "assignments_completed": ""}

    if request.method == "POST":
        values = {name: request.form.get(name, "").strip() for name in values}
        try:
            study_hours = float(values["study_hours"])
            attendance = float(values["attendance"])
            previous_score = float(values["previous_score"])
            assignments_completed = int(values["assignments_completed"])

            if not 0 <= study_hours <= 24:
                raise ValueError("Study hours must be between 0 and 24.")
            if not 0 <= attendance <= 100 or not 0 <= previous_score <= 100:
                raise ValueError("Attendance and previous score must be between 0 and 100.")
            if not 0 <= assignments_completed <= 20:
                raise ValueError("Completed assignments must be between 0 and 20.")

            sample = pd.DataFrame(
                [[study_hours, attendance, previous_score, assignments_completed]],
                columns=["study_hours", "attendance", "previous_score", "assignments_completed"],
            )
            result = MODEL.predict(sample)[0]
            confidence = max(MODEL.predict_proba(sample)[0]) * 100
            prediction = {"result": result, "confidence": round(confidence, 1)}
        except ValueError as exc:
            error = str(exc) if str(exc) else "Please enter valid numeric values."

    return render_template(
        "index.html", prediction=prediction, error=error, values=values,
        accuracy=round(MODEL_ACCURACY * 100, 1),
    )


if __name__ == "__main__":
    app.run(debug=True)
