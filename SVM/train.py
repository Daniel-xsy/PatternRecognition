from data import Dataset
from model import SVM, DualSVM
import numpy as np

def train_SVM():
    dataset = Dataset()
    x_train = dataset.x_train
    y_train = dataset.y_train
    x_test = dataset.x_test
    y_test = dataset.y_test

    model = SVM()
    epoch = 50
    lr = 0.1
    model.train(x_train, y_train, epoch=epoch, lr=lr)
    model.eval(x_test, y_test)

def qp_SVM():
    dataset = Dataset()
    x_train = dataset.x_train
    y_train = dataset.y_train
    x_test = dataset.x_test
    y_test = dataset.y_test

    model = DualSVM()
    model.quadratic_programming(x_train, y_train)
    model.eval(x_test, y_test)

if __name__=='__main__':
    qp_SVM()