# Keras-Basics
running Keras basic codes using google Colaboratory
# Predicting the diabetes disease

code file: [predictiong disease](Predicting.ipynb)

## Add more Dense layers to the existing code and check how the accuracy changes

Load the database file into the google drive colaboratory

```ruby
require 'redcarpet'
markdown = Redcarpet.new("Hello World!")
puts markdown.to_html
```

'''
from google.colab import files

uploaded = files.upload()
'''

The first line imports the keras, pandas, keras.models, and keras.layers.core modules. These modules provide a comprehensive set of tools for working with machine learning and artificial intelligence.

'''
import keras
import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
'''

 It loads the diabetes dataset into the code. The diabetes dataset is a dataset of patients with diabetes. It consists of 8 features and 1 label.

 '''
dataset = pd.read_csv('diabetes.csv', header=None).values
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, 0:8], dataset[:, 8],test_size=0.25, random_state=87)
'''

It creates a neural network model. The model has one hidden layer with 20 neurons and a ReLU activation function. The output layer has 1 neuron and a sigmoid activation function. The model is compiled with the binary_crossentropy loss function, the adam optimizer, and the acc metric. The model is then fitted to the training data for 100 epochs.

'''
np.random.seed(155)
my_first_nn = Sequential()  # create model
my_first_nn.add(Dense(20, input_dim=8, activation='relu'))  # hidden layer
my_first_nn.add(Dense(16, activation='relu'))  # additional dense layer 1
my_first_nn.add(Dense(12, activation='relu'))  # additional dense layer 2
my_first_nn.add(Dense(8, activation='relu'))  # additional dense layer 3
my_first_nn.add(Dense(1, activation='sigmoid'))  # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, initial_epoch=0)
'''

It prints a summary of the model and the results of evaluating the model on the test data.

'''
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test))
'''
## We observe that the accuracy increases when number of dense layer increases

## Change the data source to Breast Cancer dataset * available in the source code folder and make required changes. Report accuracy of the model

The first line imports the keras, pandas, keras.models, and keras.layers.core modules. These modules provide a comprehensive set of tools for working with machine learning and artificial intelligence.

'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
'''

 It loads the diabetes dataset into the code. The diabetes dataset is a dataset of patients with diabetes. It consists of 8 features and 1 label.

'''
data = pd.read_csv('breastcancer.csv', usecols=[0, 1])  # Read only the first and second columns (id and diagnosis)
dataset = data.values
'''

It encodes the diagnosis column and converts the id column to float and then it splits the data set into training data and testing data and next will reshape the input data to 2D array.

'''
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(dataset[:, 1])
Y = to_categorical(Y)


X = dataset[:, 0].astype(float)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=87)
X_train = np.reshape(X_train, (-1, 1))
X_test = np.reshape(X_test, (-1, 1))
'''

It creates a neural network model. The model has one hidden layer with 20 neurons and a ReLU activation function. The output layer has 1 neuron and a sigmoid activation function. The model is compiled with the binary_crossentropy loss function, the adam optimizer, and the acc metric. The model is then fitted to the training data for 100 epochs.

'''
np.random.seed(155)
my_first_nn = Sequential()  # Create model
my_first_nn.add(Dense(20, input_dim=1, activation='relu'))  # Hidden layer
my_first_nn.add(Dense(16, activation='relu'))  # Additional dense layer 1
my_first_nn.add(Dense(12, activation='relu'))  # Additional dense layer 2
my_first_nn.add(Dense(8, activation='relu'))  # Additional dense layer 3
my_first_nn.add(Dense(2, activation='sigmoid'))  # Output layer

my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, initial_epoch=0)
'''

It prints a summary of the model and the results of evaluating the model on the test data.

'''
print(my_first_nn.summary())
evaluation_results = my_first_nn.evaluate(X_test, Y_test)
print("Test Loss:", evaluation_results[0])
print("Test Accuracy:", evaluation_results[1])
'''

## Normalize the data before feeding the data to the model and check how the normalization change your accuracy (code given below). 

'''
from sklearn.preprocessing import 
StandardScaler sc = StandardScaler() 
'''

The first line imports the keras, pandas, keras.models, and keras.layers.core modules. These modules provide a comprehensive set of tools for working with machine learning and artificial intelligence.
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


 It loads the diabetes dataset into the code. The diabetes dataset is a dataset of patients with diabetes. It consists of 8 features and 1 label.
# Load dataset
data = pd.read_csv('breastcancer.csv', usecols=[0, 1])  # Read only the first and second columns (id and diagnosis)
dataset = data.values


It encodes the diagnosis column and normalizes the ID column and then it splits the data set into training data and testing data and next will reshape the input data to 2D array.
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(dataset[:, 1])
Y = to_categorical(Y)
# Normalize the 'id' column
sc = StandardScaler()
X = dataset[:, 0].astype(float)
X = sc.fit_transform(X.reshape(-1, 1))
X = dataset[:, 0].astype(float)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=87)
X_train = np.reshape(X_train, (-1, 1))
X_test = np.reshape(X_test, (-1, 1))


It creates a neural network model. The model has one hidden layer with 20 neurons and a ReLU activation function. The output layer has 1 neuron and a sigmoid activation function. The model is compiled with the binary_crossentropy loss function, the adam optimizer, and the acc metric. The model is then fitted to the training data for 100 epochs.
# Define the neural network model
np.random.seed(155)
my_first_nn = Sequential()  # Create model
my_first_nn.add(Dense(20, input_dim=1, activation='relu'))  # Hidden layer
my_first_nn.add(Dense(16, activation='relu'))  # Additional dense layer 1
my_first_nn.add(Dense(12, activation='relu'))  # Additional dense layer 2
my_first_nn.add(Dense(8, activation='relu'))  # Additional dense layer 3
my_first_nn.add(Dense(2, activation='sigmoid'))  # Output layer


# Compile and train the model
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, initial_epoch=0)




It prints a summary of the model and the results of evaluating the model on the test data.
# Print model summary
print(my_first_nn.summary())


# Evaluate the model and print the evaluation results (including accuracy)
evaluation_results = my_first_nn.evaluate(X_test, Y_test)
print("Test Loss:", evaluation_results[0])
print("Test Accuracy:", evaluation_results[1])


→ After normalizing there is a huge decrease in the loss and accuracy is also increased.




 Image Classification


It import the tensorflow and keras libraries, as well as the matplotlib.pyplot library. These libraries will be used to load the MNIST dataset, build a neural network, train the neural network, and plot the results.
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

It loads the MNIST dataset into the Python interpreter. The MNIST dataset is a collection of handwritten digits.

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


It normalizes the pixel values in the MNIST dataset to the range [0, 1]. This is done to improve the performance of the neural network.

# Normalize the pixel values to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0


It reshapes the images in the MNIST dataset to match the input shape of the neural network. The neural network expects the input images to be 28x28 pixel grayscale images.

# Reshape the images to match the input shape of the model
train_images = train_images.reshape(train_images.shape[0], 784)
test_images = test_images.reshape(test_images.shape[0], 784)

Plot the loss and accuracy for both training data and validation data using the history object in the source code.


The architecture of the neural network is defined. The neural network has three layers: a 64-neuron hidden layer, another 64-neuron hidden layer, and a 10-neuron output layer. The activation function for the hidden layers is relu, and the activation function for the output layer is softmax.

# Define the model architecture
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

To compile the neural network. The optimizer='adam' argument specifies that the Adam optimizer will be used, the loss='sparse_categorical_crossentropy' argument specifies that the sparse categorical crossentropy loss function will be used, and the metrics=['accuracy'] argument specifies that the accuracy metric will be used.

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

It is used to train the model

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

It plot the loss and accuracy for both training and validation data. This is done to see how the model performs
# Plot the loss and accuracy for both training and validation data
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



Plot one of the images in the test data, and then do inferencing to check what is the prediction of the model on that single image.

selects one image from the test data and makes a prediction on that image.
The image = test_images[0] line selects the first image from the test data.
The plt.imshow(image.reshape(28, 28)) line plots the image.
The prediction = model.predict(image.reshape(1, 784)) line makes a prediction on the image. The model.predict() method takes an image as input and returns a vector of probabilities. The element with the highest probability is the predicted class.
The print("Model Prediction:", prediction.argmax()) line prints the predicted class. The prediction.argmax() method returns the index of the element with the highest probability.
In this case, the predicted class is 7,


# Select one image from the test data
image = test_images[0]


# Plot the image
plt.imshow(image.reshape(28, 28))
plt.show()


# Do inference on the image
prediction = model.predict(image.reshape(1, 784))


# Print the prediction
print("Model Prediction:", prediction.argmax())


 
We had used 2 hidden layers and Relu activation. Try to change the number of hidden layer and the activation to tanh or sigmoid and see what happens. 
creates a new neural network with a different architecture than the previous neural network. The new neural network has three layers: a 128-neuron hidden layer, a 64-neuron hidden layer, and a 10-neuron output layer. The activation function for the hidden layers is tanh, and the activation function for the output layer is softmax.
The model_new.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) line compiles the new neural network. The optimizer='adam' argument specifies that the Adam optimizer will be used, the loss='sparse_categorical_crossentropy' argument specifies that the sparse categorical crossentropy loss function will be used, and the metrics=['accuracy'] argument specifies that the accuracy metric will be used.
The model_new.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels)) line trains the new neural network. The epochs=10 argument specifies that the model will be trained for 10 epochs. The validation_data=(test_images, test_labels) argument specifies that the model will be evaluated on the test data after each epoch.
The results of training the new neural network may be different from the results of training the previous neural network.


model_new = keras.Sequential([
    keras.layers.Dense(128, activation='tanh', input_shape=(784,)),
    keras.layers.Dense(64, activation='tanh'),
    keras.layers.Dense(10, activation='softmax')
])


model_new.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_new = model_new.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))


→ we observe that there is little decrease in loss and increase in accuracy.

Run the same code without scaling the images and check the performance?

It defines a neural network model with two hidden layers, trains it without scaling the input images, and evaluates its performance. The model uses the ReLU activation function for the hidden layers, softmax activation for the output layer, and sparse categorical cross-entropy loss. The training is performed for 10 epochs with the Adam optimizer. 

model_no_scaling = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


model_no_scaling.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_no_scaling = model_no_scaling.fit(train_images * 255.0, train_labels, epochs=10, validation_data=(test_images * 255.0, test_labels))




Note: The images are multiplied by 255.0 before training to rescale them from the original range of 0-1 to the range of 0-255.



→ We observe that There is decrease in loss and increase in accuracy





