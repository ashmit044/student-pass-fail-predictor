# Student Pass/Fail Predictor

A beginner-friendly command-line machine-learning project that predicts whether a student is likely to **Pass** or **Fail**. It implements the **Student Pass/Fail Prediction** project from the supplied beginner AI project list.

## What it does

The app accepts four inputs:

- Daily study hours
- Attendance percentage
- Previous score percentage
- Number of assignments completed

It passes these values to a Decision Tree classifier and prints the predicted result and the model's confidence in the terminal.

## How the code works

1. `data/student_performance.csv` contains labelled example records. Each row has the four student features and an expected result (`Pass` or `Fail`).
2. `app.py` loads the CSV using Pandas, splits the records into training and test sets, then trains a `DecisionTreeClassifier` from scikit-learn.
3. The test set is used to calculate an accuracy figure, which is displayed in the app.
4. When the user enters the values in the terminal, `app.py` validates them, creates a one-row DataFrame with the same feature columns used for training, and asks the trained model for a prediction.
5. Before using the model, the program applies a minimum academic rule: attendance below 40% or a previous score below 40% always produces `Fail`. This stops completed assignments alone from producing an unrealistic pass result.
6. The result, confidence, reason, and validation accuracy are printed in the terminal. No web interface is used.

## Project structure

```text
student-pass-fail-predictor/
├── app.py                         # Command-line ML training/prediction logic
├── requirements.txt                # Python dependencies
└── data/student_performance.csv    # Demo training data
```

## Run locally

1. Install Python 3.10 or newer.
2. In the project folder, create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies and start the program:

   ```bash
   python -m pip install -r requirements.txt
   python app.py
   ```

4. Enter the four requested values in the terminal.

You can also run one non-interactive example:

```bash
python app.py --study-hours 7 --attendance 80 --previous-score 72 --assignments-completed 9
```

## Example input

| Input | Value |
| --- | ---: |
| Daily study hours | 7 |
| Attendance | 80 |
| Previous score | 72 |
| Assignments completed | 9 |

The demo model should predict **Pass** for this profile.

## Academic minimum rule

The dataset is deliberately small and synthetic for learning purposes. To make the output sensible, a student is automatically predicted as **Fail** when either attendance or previous score is below 40%, even if they have completed many assignments. For example:

```bash
python app.py --study-hours 0 --attendance 0 --previous-score 0 --assignments-completed 10
```

This returns **Fail**.

## Notes

The included data is small and synthetic so the project is easy to understand and run. For a real use case, replace it with a larger, representative, privacy-safe dataset and assess the model for bias and reliability before using it for any decision.
