import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from libs.utils import read_to_vector, plot_stuff

exr0 = False ## Analyzing color snippets
exr1 = True ## Analyzing housing data

if exr0 :
    red_rgb, red_lab = read_to_vector(r'C:\Users\cansu\OneDrive\Desktop\PAML_2024\data\colour_snippets\red\*.png')
    grn_rgb, grn_lab = read_to_vector(r'C:\Users\cansu\OneDrive\Desktop\PAML_2024\data\colour_snippets\green\*.png')

    ## plot histograms
    bins = 100
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, tight_layout=True)
    ax[0].hist(red_rgb, bins=bins)
    ax[0].set_xlabel("Red RGB Channel Value")
    ax[0].set_ylabel("Frequency")
    ax[0].set_title("Histogram of Red RGB Channel")

    ax[1].hist(grn_rgb, bins=bins)
    ax[1].set_xlabel("Green RGB Channel Value")
    ax[1].set_ylabel("Frequency")
    ax[1].set_title("Histogram of Green RGB Channel")
    plt.show()

if exr1:
    data = pd.read_csv(r'C:\Users\cansu\OneDrive\Desktop\PAML_2024\data\housing.csv')
    print(data.keys())  # Print out the keys in the DataFrame.

    y = np.array(data['median_house_value'])

    fig_housing, axs_housing = plt.subplots(3, 3, tight_layout=True)

    for i, (key, value) in enumerate(data.items()):
        if key == "median_house_value":
            continue

        if i >= 9:
            break

        row = i // 3
        col = i % 3
        axs_housing[row, col].scatter(value, y)
        axs_housing[row, col].set_xlabel(key)
        axs_housing[row, col].set_ylabel("median_house_value")
        axs_housing[row, col].set_title(f"{key} vs. median_house_value")

    plt.show()

    x = np.array(data["median_income"])
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.5)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    ## Remove Outliers
    x_train = x_train[y_train<500000]
    y_train = y_train[y_train<500000]
    x_test = x_test[y_test<500000]
    y_test = y_test[y_test<500000]

    ## Linear Regression
    lin_reg = LinearRegression().fit(x_train.reshape(-1,1), y_train.reshape(-1,1))
    y_prediction = lin_reg.predict(x_test.reshape(-1,1)) ## x_test - predicted, y_test - ground truth
    plt.figure()
    plt.scatter(x_test,y_test) ##  scatter the data
    plt.plot( x_test, y_prediction, 'r' ) # line of regression
    plt.xlabel("median_income")
    plt.ylabel("median_house_value")
    plt.tight_layout()
    plt.show()

    ## Metrics - Mean Square Error
    mean_squared_error = mean_squared_error(y_test, y_prediction) ## y_prediction - predicted, y_test - ground truth
    print(mean_squared_error)
    print(np.mean((y_test.reshape(-1,1)-y_prediction)**2))


