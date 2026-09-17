# Titanic Survival Prediction Using Logistic Regression

This is a beginner machine learning project that uses **Logistic Regression** to predict whether a Titanic passenger survived or not.

## Dataset

The project uses the Titanic dataset.

The dataset contains information such as:

* Passenger class
* Age
* Sex
* Number of siblings/spouses
* Number of parents/children
* Fare
* Port of embarkation
* Survival status

The dataset is downloaded automatically by the Python program.

## Data Cleaning

Before training the model, the data is prepared by:

* Filling missing `Age` values with the mean age
* Removing the `Cabin` column
* Removing `Ticket`, `Name`, and `PassengerId`
* Encoding categorical variables using `pd.get_dummies()`

## Machine Learning Model

The model used in this project is:

**Logistic Regression**

The data is divided into:

* Training set
* Validation set
* Test set

The model learns from the training data and is evaluated using the validation and test data.

## Model Performance

The model achieved approximately:

* **Validation Accuracy:** 82%
* **Test Accuracy:** 81%

The model was evaluated using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score

## Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## How to Run

First, install the required libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

Then run:

```bash
python titanic.py
```

The program will automatically download the Titanic dataset and train the Logistic Regression model.

## What I Learned

Through this project, I practiced:

* Loading CSV datasets
* Data cleaning
* Handling missing values
* Removing unnecessary columns
* Encoding categorical data
* Splitting data into training, validation, and test sets
* Logistic Regression
* Making predictions
* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1-score

## Project Structure

```text
Titanic-Survival-prediction-by-using-logistic-regression/
│
├── titanic.py
├── README.md
└── .gitignore
```

## Future Improvements

Some possible improvements for this project are:

* Try Decision Tree
* Try Random Forest
* Try K-Nearest Neighbors
* Compare different machine learning models
* Perform cross-validation
* Improve feature engineering
* Tune model parameters
