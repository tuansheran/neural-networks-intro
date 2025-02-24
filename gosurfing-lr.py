#x - variables
#b- weights/co-efficents
#b0 - bias

#modle regerssion goes something like

#y = b1(x1) +  b2(x2) + b3(x3) + b4(x4) + b0
import numpy as np


#wave height
#wind speed 
#water temprature

X = np.array([
    [1.2,15,23],
    [0.8,10,24],
    [2.0,5,21],
    [1.5,12,23]
])

y_hat = np.array([[7.5], [6.0], [8.8], [7.2]])

np.random.seed(42) #fixed starting point of randomness
weights = np.random.rand(3, 1) #weights with one for each feature
bias = np.random.rand(1) #bias term

print(weights)
print(bias)

## np.dot - performs matrix calculation to 
def predict(X, weights, bias):
    return np.dot(X, weights) + bias  # y = Wx + b
    #basically doing the matrix calculation

prediction = predict(X,weights,bias)
print("Intial prediction:", prediction)
# print("Intial prediction flattern:", prediction)

#compute mean squred error - cross loss
def compute_loss(y_hat, y_predicted):
    return np.mean((y_hat - y_predicted) ** 2) #MSE formula

loss = compute_loss(y_hat,prediction)
print("Intial loss", loss) #identifies how far away the value is from the original dot


#add the optimizer 
def compute_gradients(y_hat, y_predicted):
    n = len(y_hat)
    gradient_of_weights = (-2/n) * np.dot(X.T, (y_hat - y_predicted)) #tells how much of weight contribute to the error
    gradient_of_bias = (-2/n) * np.sum(y_hat - y_predicted) #tells how much of bias contribute to the error
    return gradient_of_weights, gradient_of_bias #return the contribution of erros

def update_weights(weights, bias, gW, gB, learning_rate):
    weights -= learning_rate * gW 
    bias -= learning_rate * gB
    return weights, bias

learning_rate = 0.0001
epochs = 100

#leanring rate 0.001 2 epoch 32.

for epoch in range(epochs):
    predictions = predict(X,weights,bias)
    loss = compute_loss(y_hat, predictions)
    dW, dB = compute_gradients(y_hat, prediction)
    weights, bias = update_weights(weights, bias, dW, dB, learning_rate)

    print(f"Epoch {epoch}, loss {loss}")

    
    




     

