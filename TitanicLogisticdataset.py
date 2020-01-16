import math
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def TitanicLogistic():
    # step1: Load data
    titanic_data = pd.read_csv('TitanicDataset.csv')

    print("First 5 entries from loaded dataset")
    print(titanic_data.head())

    print("Number of passangers are " + str(len(titanic_data)))

    # Step 2: Analyze data
    print("Visualisation : Survived and non survied passangers")
    figure()
    target = "Survived"

    countplot(data=titanic_data, x=target).set_title("Survived and non survied passangers")
    show()

    print("Visualisation: Survived and non survied passangers based on Gender")
    figure()
    target = "Survived"

    countplot(data=titanic_data, x=target, hue="Sex").set_title("Survived and non survied passangers based on Gender")
    show()

    print("Visualisation : Survived and non survied passangers based on the passanger class")
    figure()
    target = "Survived"

    countplot(data=titanic_data, x=target, hue="Pclass").set_title(
        "Survived and non survied passangers based on the Passanger class")
    show()

    print("Visualisation : Survived and non survied passangers based on Age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Survived and non survied passangers based on Age")
    show()

    print("Survived and non survied passangers based on the Fare")
    figure()
    titanic_data["Fare"].plot.hist().set_title("Survived and non survied passangers based on Fare")
    show()

    # Step 3 : Data Cleaning
    titanic_data.drop("zero",axis=1,inplace=True)

    print("First 5 entries from loaded dataset after removing zero column")
    print(titanic_data.head(5))

    print("values of Sex column")
    print(pd.get_dummies(titanic_data["Sex"]))

    print("Values of sex column after removing one field")
    Sex=pd.get_dummies(titanic_data["Sex"],drop_first=True)
    print(Sex.head(5))
    

def main():
    print("Supervised Machine Learning")

    print("Logistic regression on Titanic data set")

    TitanicLogistic()


if __name__ == "__main__":
    main()
