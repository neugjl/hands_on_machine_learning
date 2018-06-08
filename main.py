import numpy as np

N, D_in, H, D_out = 64, 1000, 100,10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H,D_out)

learning_rate = 1e-6
for t in range(50):
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    loss = np.square(y_pred-y).sum()
    print(t,loss)

    grad_y_pred = 2.0 * (y_pred-y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2



template = '{0:0.2f} {1:s} are worth US${2:d}'
result=template.format(4.5560,'Argentine Pesos',1)
strings = ['a','as','bat','car','dove','python']
lengths = [len(x) for x in strings]
unique_lenghts = {len(x) for x in strings}
unique_lenghts2 = set(lengths)

a = []
def func():
    global a
    a = None
func()

import numpy as np
b= [1,2,2,3,3,4,5]
c= b[2:4]
c[1] = 10
a_b = np.array([1,2,2,3,3,4,5])
a_c = a_b[2:4]
a_c[1] = 10

import pandas as pd
obj = pd.Series([4,7,-5,3],index=('a','b','c','d'))

import numpy as np
from matplotlib import pyplot as plt
X = np.linspace(-np.pi,np.pi,256,endpoint=True)
C = np.cos(X)
S = np.sin(X)
plt.plot(X,C)
plt.plot(X,S)
plt.show()
