import numpy as np 
import matplotlib.pyplot as plt

def plot_decision_boundary(x, pred_func):  

    ## 设定最大最小值，边缘填充  
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5  
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5  
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  

    ## 预测 
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])  
    Z = Z.reshape(xx.shape)  

    plt.contourf(xx, yy, Z)