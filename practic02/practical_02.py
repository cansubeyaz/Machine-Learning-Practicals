import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

print(sys.path)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from Lib.utils import read_to_vector, plot_stuff

exr0 = True
exr1 = False

if exr0:
    red_rgb, red_lab = read_to_vector('data/colour_snippets/red/*.png')
    grn_rgb, grn_lab = read_to_vector('data/colour_snippets/green/*.png')

    bins = 100
    # fig, ax = plt.subplots( 2, 1, sharex=True, tight_layout=True )
    # ax[0].hist( red_rgb, bins=bins )
    # ax[1].hist( grn_rgb, bins=bins )
    # plt.show()
    #
    # fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True)
    # ax[0].hist(red_lab, bins=bins)
    # ax[1].hist(grn_lab, bins=bins)
    # plt.show()
    plot_stuff(bins, red_rgb, grn_rgb)
    plot_stuff(bins, red_lab, grn_lab)

if exr1:
    data = pd.read_csv('data/housing.csv')
    if False:
        for k in data.keys():
            print(k)
    y = np.array(data['median_house_value'])
    if False:
        for k, v in data.items():
            if k == 'median_house_value':
                continue
            plt.figure()
            plt.scatter(v, y)
            plt.xlabel(k)
            plt.ylabel('median_house_value')
            plt.tight_layout()
            plt.show()
    x = np.array(data['median_income'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    x_train = x_train[y_train < 500000]
    y_train = y_train[y_train < 500000]
    x_test = x_test[y_test < 500000]
    y_test = y_test[y_test < 500000]

    plt.figure()
    plt.scatter(x_train, y_train)
    plt.xlabel('median_income')
    plt.ylabel('median_house_value')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(x_test, y_test)
    plt.xlabel('median_income')
    plt.ylabel('median_house_value')
    plt.tight_layout()
    plt.show()

    linreg = LinearRegression().fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
    y_pred = linreg.predict(x_test.reshape(-1, 1))
    plt.figure()
    plt.scatter(x_test, y_test)
    plt.plot(x_test, y_pred, 'r')
    plt.xlabel('median_income')
    plt.ylabel('median_house_value')
    plt.tight_layout()
    plt.show()

    mse = mean_squared_error(y_test, y_pred)
    print(mse)
    print(np.mean((y_test.reshape(-1, 1) - y_pred) ** 2))
