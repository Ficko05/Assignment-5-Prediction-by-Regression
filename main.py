import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split


def setup():
    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', 20)


def read_data():
    url = "./data/china_gdp.csv"
    df = pd.read_csv(url)
    #print(df.head())
    return df


def plot_diagrams(df):
    plt.ylabel('years')
    plt.xlabel('value')
    plt.scatter(df.Value, df.Year, color='green')
    plt.show()

    sns.distplot(df['Value'], label='Value', norm_hist=True)
    plt.show()


def data_prep(df):
    corr_matrix = df.corr()
    print(corr_matrix)


def train_model_linear_regression(df):
    # split x and y
    X = df.iloc[:, 0:1].values
    y = df.iloc[:, 1].values
    # Split into the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # creating Linear Regression model
    linreg = LinearRegression()
    # fitting the model to our data
    linreg.fit(X, y)
    y_predicted = linreg.predict(X)
    #print(y_predicted)
    print("---------------------------")
    #print(y)

    # Visualise the Linear Regression
    plt.title('Linear Regression')
    plt.scatter(X, y, color='red')
    plt.plot(X, y_predicted, color='blue')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.show()
    print("-----linear-----")

    # Predicting a new result with Linear Regression
    print(float(linreg.predict([[2010]])))
    print(float(linreg.predict([[2020]])))
    print(float(linreg.predict([[2022]])))


def train_model_ploynomial_regression(df):
    # split x and y
    X = df.iloc[:, 0:1].values
    y = df.iloc[:, 1].values
    # Split into the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # polynomial regression model
    polyreg = PolynomialFeatures(degree=4)
    # transform my train data to adjust the polynom to linear regression model
    X_pol = polyreg.fit_transform(X)
    # create linear regression model
    pollinreg = LinearRegression()
    pollinreg.fit(X_pol, y)
    # apply the model on my train data
    y_predicted = pollinreg.predict(X_pol)
    print("-----poly-----")
    #print("-----y_predicted-----")
    #print(y_predicted)
    #print("-----Y-----")
    #print(y)


    # Visualise the Polymonial Regression results
    plt.title('Polynomial Regression')
    plt.scatter(X, y, color='red')
    plt.plot(X, y_predicted, color='green')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.show()

    print("Multiple Regression Performance")
    # The coefficients
    print('Coefficient: ', pollinreg.coef_)
    print('Intercept: ', pollinreg.intercept_)
    print("-------------")

    # Predicting the same with Polymonial Regression
    print(float(pollinreg.predict(polyreg.fit_transform([[2010]]))))
    print(float(pollinreg.predict(polyreg.fit_transform([[2020]]))))
    print(float(pollinreg.predict(polyreg.fit_transform([[2022]]))))
    print("-----------")

    # RMSE (Root mean squared error) answers the question: "How similar, on average, are the numbers in list1 to list2?"
    rmse = np.sqrt(sm.mean_squared_error(y, y_predicted))
    r2 = sm.r2_score(y, y_predicted)
    print(rmse)
    print(r2)



if __name__ == '__main__':
    setup()
    df = read_data()
    print(df.shape)
    plot_diagrams(df)
    data_prep(df)
    train_model_linear_regression(df)
    train_model_ploynomial_regression(df)