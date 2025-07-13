# 🚢 Titanic Survival Prediction

A machine learning project that predicts whether a passenger would survive the Titanic disaster, based on passenger information.  
This is a classic classification problem using the Titanic dataset from [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset).

##  Model Overview

This project uses a **Random Forest Classifier** trained on selected features such as:
- Passenger class
- Sex
- Age
- Family members aboard (siblings/spouse, parents/children)
- Fare
- Embarked location
- Deck & Title (engineered from Name and Cabin)

##  Dataset

The dataset contains the following key features:

| Feature     | Description                            |
|-------------|----------------------------------------|
| `Survived`  | 0 = No, 1 = Yes                        |
| `Pclass`    | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)|
| `Sex`       | Gender                                 |
| `Age`       | Age in years                           |
| `SibSp`     | Siblings/Spouses aboard                |
| `Parch`     | Parents/Children aboard                |
| `Fare`      | Ticket price                           |
| `Embarked`  | Port of Embarkation (C, Q, S)          |

Additional features like `Deck` and `Title` were extracted and encoded.

##  Technologies Used

- **Python 3**
- **Pandas**, **NumPy** for data manipulation
- **Scikit-learn** for modeling and preprocessing
- **Streamlit** for the web interface
- **Matplotlib/Seaborn** (optional) for EDA and visualizations

##  Model Performance

| Model              | Accuracy |
|--------------------|----------|
| Logistic Regression| 79%      |
| KNeighborsClassifier| 79%      |
| MLPClassifier      | 77%      |
| Random Forest (tuned with GridSearchCV) | **83%**  |

##  Streamlit Web App

A sleek, interactive Streamlit UI has been built with:
- A Titanic-themed background
- Stylish form inputs
- Emoji-based survival results (✅ or ❌)

###  Prediction Output
- ✅ **Green success box** if the passenger would survive
- ❌ **Red danger box** if not

##  Demo Video

📂 [Click here to view the demo video](./TitanicSurvivalPrediction._demo.mp4)

##  Customizations
- Background image with frosted-glass effect
- Styled sliders, dropdowns, and labels
- Uses joblib for model persistence


