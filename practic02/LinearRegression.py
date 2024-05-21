import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

x = [5,7,8,7,2,17,2,9,4,11,12,9,6] ## age - independent
y = [99,86,87,88,111,86,103,87,94,78,77,85,86] ## speed - dependent

slope, intercept, r, p, std_err = stats.linregress(x,y) ## some important key values of Linear Regression

def myfunc(x):
    return slope*x + intercept
mymodel = list(map(myfunc,x)) ## Run each value of the x array through the function

plt.scatter(x,y) ## data points x and y
plt.plot(x, mymodel, "r") ##line of regression
plt.tight_layout()
plt.xlabel("Age")
plt.ylabel("Speed")
plt.title("Linear Regression")
plt.show()

############
x1 = np.array([5,15,25,35,45,55]).reshape(-1,1) ## input - x.shape is (6, 1) 2 dimensions
y1 = np.array([5,20,14,32,22,38]) ## output -  y.shape is (6,) single dimensions

print("x1",x1)
print("y1",y1)

model = LinearRegression()
model_fit = model.fit(x1,y1)
model_score = model.score(x1,y1)
print("Model score :", model_score) ##
print("Intercept :",{model.intercept_}) ## ax + b, b - intercept

new_model = model.fit(x1, y1.reshape(-1,1)) ## fit the data X1 TRAİN RGB İNPUT, Y1 TRAİN OUTPUT LABELS
y1_predict = model.predict(x1) ## Predict --- y_predict = model.intercept_ + model.coef_ * x

plt.scatter(x1,y1)
plt.plot(x1, y1_predict,"g")
plt.tight_layout()
plt.xlabel("x1")
plt.ylabel("y1")
plt.title("Linear Regression")
plt.show()