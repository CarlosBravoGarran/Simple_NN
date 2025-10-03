import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data: divide by 255 so values are between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the image (28x28) to a vector
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU
    Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each class)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Make a prediction
predictions = model.predict(x_test)

# Show an example image and its prediction
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.title(f"Prediction: {predictions[0].argmax()}")
plt.show()
