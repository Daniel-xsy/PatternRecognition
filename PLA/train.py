from matplotlib.pyplot import show
from data import PointData
from pla import PLA,Pocket
import numpy as np


train_num = 10
test_num = 4

dataset = PointData(train_num, test_num, linear=False)
x_train = dataset.x_train
y_train = dataset.y_train
x_test = dataset.x_test
y_test = dataset.y_test

x=[
   0.2, 0.7,
   0.3, 0.3,
   0.4, 0.5,
   0.6, 0.5,
   0.1, 0.4,
   0.4, 0.6,
   0.6, 0.2,
   0.7, 0.4,
   0.8, 0.6,
   0.7, 0.5]
x = np.array(x).reshape(-1, 2)

y=[1, 1, 1, 1, 1,
    -1,-1,-1,-1,-1]
y = np.array(y).reshape(-1, 1)

# PLA
# classifier = PLA(dimension=2)
# classifier.train(x_train,y_train)
# classifier.inference(x_test,y_test)

# Pocket
classifier = Pocket(dimension=2)
classifier.train(x, y, show=True)
