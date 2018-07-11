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

from scipy import io as spio
import numpy as np
a = np.ones((3,3))
spio.savemat('file.mat',{'a':a})
data = spio.loadmat('file.mat')

from scipy.integrate import quad
res, err = quad(np.sin,0,np.pi/2)
check = np.allclose(res,1)

import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt
time_step = 0.02
period = 5
time_vec =  np.arange(0,20,time_step)
sig = (np.sin(2*np.pi/period*time_vec)+0.5*np.random.rand(time_vec.size))
plt.figure(figsize=(6,5))
plt.plot(time_vec,sig,label="Original signal")
plt.show()
sig_fft = fftpack.fft(sig)
power = np.abs(sig_fft)
sample_freq = fftpack.fftfreq(sig.size, d=time_step)
plt.figure(figsize=(6, 5))
plt.plot(sample_freq, power)
plt.xlabel("Frequency[Hz]")
plt.ylabel("power")
#Find the peak frequency, we can focus on only the positive frequencies
pos_mask = np.where(sample_freq > 0)
freq = sample_freq[pos_mask]
peak_freq = freq[power[pos_mask].argmax()]
#check that it does indeed correspond to the frequency that we generate the signal with
np.allclose(peak_freq,1./period)
#An inner plot the show the peak frequency
axes = plt.axes([0.55,0.3,0.3,0.5])
plt.title("Peak Frequency")
plt.plot(freq[:8],power[:8])
plt.setp(axes,yticks=[])
high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq)> peak_freq]=0
filtered_sig = fftpack.ifft(high_freq_fft)
plt.figure(figsize=(6,5))
plt.plot(time_vec,sig,label='Original signal')
plt.plot(time_vec,filtered_sig,linewidth=3,label='FilteredOriginal signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend(loc='best')

num = [1,2,3]
num = iter(num)
num.__iter__()

import numpy as np
x = np.array([1,2,3],dtype=np.int32)
x.data
bytes(x.data)
np.dtype(int).type

from sklearn import neighbors,datasets
iris = datasets.load_iris()
X,y = iris.data,iris.target
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)
print(iris.target_names[knn.predict([[3,5,4,2]])])
knn.score(X,y)

from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
digits = load_digits()
X_train, X_test, y_train,y_test = train_test_split(digits.data,digits.target)
clf = GaussianNB()
clf.fit(X_train,y_train)
predict = clf.predict(X_test)
expected = y_test
match = (expected == predict)
print(match.sum()/len(match))
print(metrics.classification_report(expected,predict))
print(metrics.confusion_matrix(expected,predict))

from urllib.request import urlopen
html = urlopen('http://pythonscraping.com/pages/page1.html')
print(html.read())

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X,y = mnist["data"],mnist["target"]
X.shape
y.shape
import matplotlib
import matplotlib.pyplot as plt
some_digit = X[36000]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
y[36000]

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def plot_learning_curve(model,X,y):
    fig = plt.figure()
    X_train, X_val,y_train,y_val = train_test_split(X,y,test_size=0.2)
    train_errors,val_errors = [],[]
    for m in range(1,len(X_train)):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict,y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict,y_val))
    plt.plot(np.square(train_errors),"r-+",linewidth=2,label="train")
    plt.plot(np.square(val_errors),"b-",linewidth=3,label="val")

polynominal_regression = Pipeline((
    ("poly_features",PolynomialFeatures(degree=10,include_bias=False)),
    ("sgd_ref",LinearRegression())
))
m=100
X = 6*np.random.rand(m,1)-3
y = 0.5 * X**2 + X + 2 + np.random.randn(m,1)
poly_features = PolynomialFeatures(degree = 2,include_bias=False)
X_poly = poly_features.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)
plot_learning_curve(lin_reg,X,y)
plot_learning_curve(polynominal_regression,X,y)
fig = plt.figure()
plt.scatter(X,y)

#Central Limit Theorem
import numpy as np
import matplotlib.pyplot as plt
sample = 100000
x_rand = np.zeros(sample)
for t in range(3):
    x_rand += np.random.rand(sample)
x_randn = np.random.randn(sample)
#y = np.arange(0,x.size,1)
#plt.scatter(y,x)
#plt.hist(x_randn,bins=100)
plt.hist(x_rand,bins=100)
