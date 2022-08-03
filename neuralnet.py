import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

train_data = pd.read_csv('datasets/train.csv')
test_data = pd.read_csv('datasets/test.csv')

train_data = np.array(train_data)
test_data = np.array(test_data)

m,n = train_data.shape #m=42.000 n=785
y_train = train_data.T[0]
x_train = train_data.T[1:n] / 255

x_test = test_data.T / 255

#now we have 42.000 training samples and 28.000 test samples 


def initialize_parameters():
    w1 = np.random.randn(16,784)
    b1 = np.random.randn(16,1)
    w2 = np.random.randn(16,16)
    b2 = np.random.randn(16,1)
    w3 = np.random.randn(10,16)
    b3 = np.random.randn(10,1)
    return w3,b3,w2,b2,w1,b1
def softmax(z):
    exp = np.exp(z - np.max(z)) 
    return exp / exp.sum(axis=0,keepdims=True) + (1/100000000)

def leaky_relu(x):
    return np.where(x > 0, x, x * 0.05) 

def derivative_of_leaky_relu(x):
    return np.where(x >= 0, 1, 0.05)

def forward_propagation(x,w3,b3,w2,b2,w1,b1):
    z1 = np.dot(w1,x) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(w2,a1) + b2
    a2 = leaky_relu(z2)
    z3 = np.dot(w3,a2) + b3
    a3 = softmax(z3)
    return z3,a3,z2,a2,z1,a1

def one_hot_encoding(y):
    one_hot_y = np.zeros((y.size,10))
    one_hot_y[np.arange(y.size),y] = 1
    return one_hot_y.T

def cost_function(y,al):
    return -np.sum(y * np.log(al + 0.00000001) + (1-y) * np.log(1-al + 0.00000001),axis=1,keepdims=True) / m

def back_propagation(x,y,z3,a3,w3,z2,a2,w2,z1,a1,w1):
    one_hot = one_hot_encoding(y)
    s3 = a3 - one_hot
    dw3 = np.dot(s3,a2.T) / m
    db3 = np.sum(s3,axis=1,keepdims=True) / m 
    s2 = np.dot(w3.T,s3) * derivative_of_leaky_relu(z2)
    dw2 = np.dot(s2,a1.T) / m
    db2 = np.sum(s2,axis=1,keepdims=True) / m
    s1 = np.dot(w2.T,s2) * derivative_of_leaky_relu(z1)
    dw1 = np.dot(s1,x.T) / m
    db1 = np.sum(s1,axis=1,keepdims=True) / m
    return dw3,db3,dw2,db2,dw1,db1

def parameter_update(w3, b3, w2, b2,w1,b1, dw3, db3, dw2, db2,dw1,db1, alpha):
    w3 -= alpha * dw3  
    b3 -= b3 - alpha * db3    
    w2 -= alpha * dw2  
    b2 -= alpha * db2
    w1 -= alpha * dw1
    b1 -= alpha * db1
    return w3, b3, w2, b2, w1, b1

def get_predictions(a3):
    return np.argmax(a3, 0)

def get_accuracy(predictions, actual):
    return np.sum(predictions == actual) / actual.size

def gradient_descent(alpha,iteration):
    w3,b3,w2,b2,w1,b1 = initialize_parameters()
    for i in range(0,iteration):
        z3,a3,z2,a2,z1,a1 = forward_propagation(x_train,w3,b3,w2,b2,w1,b1)
        dw3,db3,dw2,db2,dw1,db1 = back_propagation(x_train,y_train,z3,a3,w3,z2,a2,w2,z1,a1,w1)
        w3,b3,w2,b2,w1,b1 = parameter_update(w3, b3, w2, b2,w1,b1, dw3, db3, dw2, db2,dw1,db1, alpha)
        if (i+1) % 20 == 0:
            print(f"Iteration: {i+1} / {iteration}")
            prediction = get_predictions(a3)
            print(f'{get_accuracy(prediction, y_train):.3%}')
            print(a3[:,1])
            #print("cost:{}".format(np.sum(cost_function(y_train,a3))/10))
    return w3,b3,w2,b2,w1,b1

def main():
    #plt.gray()
    #plt.imshow(x_train[:,1].reshape((28,28)), interpolation='nearest')
    #plt.show()
    #x = np.array([[-5,5,5],[2,5,3],[-3,5,-7],[2,5,-1]])
    w3,b3,w2,b2,w1,b1 = gradient_descent(0.1,100)
    
    
if __name__ == "__main__":
    main()
        