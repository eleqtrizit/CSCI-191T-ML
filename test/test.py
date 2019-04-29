import numpy as np
import math


def sigmoid(z):
    return 1.0/(1+math.exp(-z))

# estimated probability that y=1 on input x


def hypothesis(ws, x):
    return sigmoid(sum(np.dot(ws, x)))


def costFunctionLogReg(theta):
    if (y == 1):
        -1 * math.log(hypothesis(x))
    else:
        -1 * math.log(1 - hypothesis(x))


def logit(x):
    odds = sigmoid(x) / (1 - sigmoid(x))


def slog(x):
    return 0 if x == 0 else math.log(x)


def crossEntropy(features, targets, weights):
    ce = 0
    for x, t in zip(features, targets):
        h = sigmoid(sum(weights*x))
        ce = ce + -1*(t*slog(h) + (1-t)*slog(1-h))
    return ce


features = np.array([[1, 2, 3], [1, 3, 4], [1, 4, 5],
                     [1, 5, 6], [1, 6, 7], [1, 8, 9]])
targets = np.array([0, 0, 0, 1, 1, 1])
weights = np.array([0, 0, 0])
g = np.array([0, 0, 0])
e = 0
rate = 0.020

e = crossEntropy(features, targets, weights)
print(e)
