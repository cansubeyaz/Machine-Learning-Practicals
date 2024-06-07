import openml as oml
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from random import randint  # Ensure randint is imported

# Download FMNIST data
mnist = oml.datasets.get_dataset(40996)
X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute, dataset_format='array')
X = X.reshape(70000, 28, 28)
fmnist_classes = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}

# Plot random samples from the dataset
fig, axes = plt.subplots(1, 5, figsize=(10, 5))
for i in range(5):
    n = randint(0, 70000)
    axes[i].imshow(X[n], cmap=plt.cm.gray_r)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_xlabel("{}".format(fmnist_classes[y[n]]))
plt.show()

# Preprocess the data
Xf_train, Xf_test, yf_train, yf_test = train_test_split(X, y, train_size=60000, shuffle=True, random_state=0)
Xf_train = Xf_train.reshape((60000, 28 * 28)).astype('float32') / 255
Xf_test = Xf_test.reshape((10000, 28 * 28)).astype('float32') / 255
yf_train = to_categorical(yf_train)
yf_test = to_categorical(yf_test)

# Define the neural network
network = models.Sequential()
network.add(layers.Input(shape=(28 * 28,)))
network.add(layers.Dense(512, activation='relu', kernel_initializer='he_normal'))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(512, activation='relu', kernel_initializer='he_normal'))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(10, activation='softmax'))
network.summary()

# Compile the neural network
network.compile(
    loss=CategoricalCrossentropy(label_smoothing=0.01),
    optimizer=RMSprop(learning_rate=0.001, momentum=0.0),
    metrics=[CategoricalAccuracy()]
)

# Train the neural network
history = network.fit(Xf_train, yf_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2)

# Evaluate the neural network on test data
test_loss, test_acc = network.evaluate(Xf_test, yf_test, verbose=0)
print('Test accuracy:', test_acc)

# Visualize a sample prediction
np.set_printoptions(precision=7)
# Sample IDd
sample_id = 4

# Visualize the sample prediction
fig, axes = plt.subplots(1, 1, figsize=(2, 2))
axes.imshow(Xf_test[sample_id].reshape(28, 28), cmap=plt.cm.gray_r)
axes.set_xlabel("True label: {}".format(fmnist_classes[np.argmax(yf_test[sample_id])]))
axes.set_xticks([])
axes.set_yticks([])
plt.show()

# Print the prediction of the sample
sample_prediction = network.predict(Xf_test, verbose=0)[sample_id]
print(sample_prediction)

