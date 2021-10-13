from data import Dataset
from model import LogisticRegression
import numpy as np

data2 = Dataset()
x_train = data2.x_train
y_train = data2.y_train
x_test = data2.x_test
y_test = data2.y_test

epoch = 50
lr = 0.01

model = LogisticRegression(lr=lr, epoch=epoch)
yhat, loss = model(x_train, y_train)

model.eval(x_test,y_test)