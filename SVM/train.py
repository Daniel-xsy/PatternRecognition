from data import Dataset
from model import SVM, DualSVM, KernelSVM
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

def qp_SVM(mode = 'dualsvm'):
    dataset = Dataset()
    x_train = dataset.x_train
    y_train = dataset.y_train
    x_test = dataset.x_test
    y_test = dataset.y_test

    if mode == 'svm':
        model = SVM(dimension=2)

    elif mode == 'dualsvm':
        model = DualSVM(dimension=2)

    elif mode == 'kernelsvm':
        model = KernelSVM(nonlinear=2, zeta=1, gamma=1, gauss=False)

    elif mode == 'gausskernel':
        model = KernelSVM(nonlinear=2, zeta=1, gamma=1, gauss=True)

    else:
        raise NotImplemented

    model.quadratic_programming(x_train, y_train)
    model.eval(x_test, y_test)

if __name__=='__main__':
    mode = ['svm','dualsvm','kernelsvm','gausskernel']
    qp_SVM(mode=mode[3])