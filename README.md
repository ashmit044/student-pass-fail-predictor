# Student Pass/Fail Predictor

A beginner-friendly machine-learning web app that predicts whether a student is likely to **Pass** or **Fail**. It implements the **Student Pass/Fail Prediction** project from the supplied beginner AI project list.

## What it does

The app accepts four inputs:

- Daily study hours
- Attendance percentage
- Previous score percentage
- Number of assignments completed

It passes these values to a Decision Tree classifier and displays the predicted result and the model's confidence.

## How the code works

1. `data/student_performance.csv` contains labelled example records. Each row has the four student features and an expected result (`Pass` or `Fail`).
2. `app.py` loads the CSV using Pandas, splits the records into training and test sets, then trains a `DecisionTreeClassifier` from scikit-learn.
3. The test set is used to calculate an accuracy figure, which is displayed in the app.
4. When the form is submitted, Flask validates the entered values, creates a one-row DataFrame with the same feature columns used for training, and asks the trained model for a prediction.
5. Flask renders `templates/index.html`; `static/style.css` provides the responsive layout.

## Project structure

```text
student-pass-fail-predictor/
├── app.py                         # Flask app and ML training/prediction logic
├── requirements.txt                # Python dependencies
├── data/student_performance.csv    # Demo training data
├── templates/index.html            # Web interface
└── static/style.css                # Styling
```

## Run locally

1. Install Python 3.10 or newer.
2. In the project folder, create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies and start the app:

   ```bash
   pip install -r requirements.txt
   python app.py
   ```

4. Open `http://127.0.0.1:5000` in your browser.

## Example input

| Input | Value |
| --- | ---: |
| Daily study hours | 7 |
| Attendance | 80 |
| Previous score | 72 |
| Assignments completed | 9 |

The demo model should predict **Pass** for this profile.

## Notes

The included data is small and synthetic so the project is easy to understand and run. For a real use case, replace it with a larger, representative, privacy-safe dataset and assess the model for bias and reliability before using it for any decision.
