import sys, os
import numpy as np

learning_rate = 0.1

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    x = x - np.max(x)
    sum_ex = np.sum(np.exp(x))
    # print(sum_ex)
    return np.exp(x) / sum_ex

# def numerical_gradient(f, x):
#     h = 1e-4 # 0.0001
#     grad = np.zeros_like(x)
    
#     it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
#     while not it.finished:
#         idx = it.multi_index
#         tmp_val = x[idx]
#         x[idx] = float(tmp_val) + h
#         fxh1 = f(x) # f(x+h)
        
#         x[idx] = tmp_val - h 
#         fxh2 = f(x) # f(x-h)
#         grad[idx] = (fxh1 - fxh2) / (2*h)
        
#         x[idx] = tmp_val # 还原值
#         it.iternext()   
        
#     return grad

class network:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.params['B1'] = np.zeros((1, hidden_size))
        self.params['B2'] = np.zeros((1, output_size))
        # for i in self.params:
        #     print(i + ": ")
        #     print(self.params[i].shape)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        B1, B2 = self.params['B1'], self.params['B2']

        h1 = np.dot(x, W1) + B1
        sh1 = sigmoid(h1)

        h2 = np.dot(sh1, W2) + B2
        sh2 = softmax(h2)

        # print(h2)
        # print(sh2)

        return sh2

    # 抽象的数值近似梯度计算
    # def numerical_gradient(self, x, t):
    #     loss_W = lambda W: self.loss(x, t)

    #     grad = {}
    #     grad['W1'] = numerical_gradient(loss_W, self.params['W1'])
    #     grad['W2'] = numerical_gradient(loss_W, self.params['W2'])
    #     grad['B1'] = numerical_gradient(loss_W, self.params['B1'])
    #     grad['B2'] = numerical_gradient(loss_W, self.params['B2'])
    #     return grad

    # 真神：反向传播
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        B1, B2 = self.params['B1'], self.params['B2']
        h1 = np.dot(x, W1) + B1
        sh1 = sigmoid(h1)
        h2 = np.dot(sh1, W2) + B2
        sh2 = softmax(h2)

        # 后半截
        dL_dh2 = sh2 - t                # (1, 10)
        dL_dB2 = dL_dh2                 # (1, 10)
        # sh1: (1, 50), dL_dB2: (1, 10)
        dL_dW2 = np.dot(sh1.T, dL_dB2)  # (50, 10)
        # dL_dB2: (1, 10), W2: 50, 10
        dL_dsh1 = np.dot(dL_dB2, W2.T)  # (1, 50)

        # 前半截
        dh1_dB1 = 1                     # (1, 1)
        dh1_dW1 = x.T                   # (784, 1)
        dsh1_dh1 = sh1 * (1 - sh1)      # (1, 50)

        #dL_dsh1: (1, 50), dsh1_dh1: (1, 50)
        dL_dh1 = dL_dsh1 * dsh1_dh1     # (1, 50)
        dL_dB1 = dL_dh1 * dh1_dB1       # (1, 50)
        #dh1_dW1: (784, 1), dL_dh1: (1, 50)
        dL_dW1 = np.dot(dh1_dW1, dL_dh1)# (784, 50)

        grad = {}
        grad['W1'] = dL_dW1
        grad['W2'] = dL_dW2
        grad['B1'] = dL_dB1
        grad['B2'] = dL_dB2

        # print("grident done")
        return grad

    # def gradient2(self, x, t):
    #     W1, W2 = self.params['W1'], self.params['W2']
    #     B1, B2 = self.params['B1'], self.params['B2']
    #     grads = {}
        
    #     batch_num = x.shape[0]
        
    #     # forward
    #     a1 = np.dot(x, W1) + B1
    #     z1 = sigmoid(a1)
    #     a2 = np.dot(z1, W2) + B2
    #     y = softmax(a2)
        
    #     # backward
    #     dy = (y - t) / batch_num
    #     grads['W2'] = np.dot(z1.T, dy)
    #     grads['B2'] = np.sum(dy, axis=0)
        
    #     da1 = np.dot(dy, W2.T)
    #     dz1 = (1.0 - sigmoid(a1)) * sigmoid(a1) * da1
    #     grads['W1'] = np.dot(x.T, dz1)
    #     grads['B1'] = np.sum(dz1, axis=0)

    #     return grads

    def loss(self, x, t):
        output = self.predict(x)
        return -np.log(output[0][t] + 1e-7)
    
    def accuracy(self, x, t):
        output = self.predict(x)
        return int(np.argmax(output[0]) == t)

    def update(self, grad):
        for k in grad:
            self.params[k] -= learning_rate * grad[k]