import numpy as np
from numpy.lib.shape_base import expand_dims
from cvxopt.solvers import qp
from cvxopt import matrix

def hinge_loss(y, yhat):
    batch_size = y.shape[0]

    loss = 1-y*yhat
    loss = np.clip(loss, a_min=0, a_max=None)

    return np.sum(loss,axis=0)/batch_size

class SVM:
    def __init__(self, dimension=2):
        self.dimension = dimension
        self.w = np.zeros((dimension+1,1))  ## 增广化

    def quadratic_programming(self, x, y):
        ## 使用二次规划求解
        n, dimension = x.shape
        if not dimension == self.dimension:
            raise

        Q = np.eye(self.dimension+1)
        Q[0][0] = 0

        p = np.zeros((self.dimension+1,1))
        expand_dims = np.ones((n,1))
        A = -y * np.concatenate((expand_dims, x), axis=1)
        c = -1 * np.ones((n,1))

        Q = matrix(Q)
        p = matrix(p)
        A = matrix(A)
        c = matrix(c)

        slv = qp(Q, p, A, c)['x']
        self.w = np.array(slv).reshape(dimension+1,1)

    def train(self, x, y, epoch=10, lr=0.1):
        ## 梯度下降求解
        '''
        Args: 
             x : `array_like` (n, dimension)
             y : `array_like` (n, 1)
             mode: 'qp': quadratic programming
                   'gd': gradient descent
        '''
        n, dimension = x.shape
        if not dimension == self.dimension:
            raise
        expand_dims = np.ones((n,1))
        x = np.concatenate((expand_dims, x), axis=1) ## 增广化
        
        losses = []
        for n_epoch in range(epoch):

            yhat = np.matmul(x,self.w)
            loss = hinge_loss(y, yhat)
            losses.append(loss)
            grad = self._caculate_grad(x, y, yhat)
            self.w -= lr * grad

            print('epoch: %i/%i  loss: %.2f'%(n_epoch+1, epoch, loss))

        return losses


    def _caculate_grad(self, x, y, yhat):
        index = np.where((1 - y * yhat) < 0)[0]
        grad = -y * x
        grad[index] = 0
        grad = np.sum(grad, axis=0)

        return grad.reshape(self.dimension+1, 1)

    def eval(self, x, y):
        n, dimension = x.shape
        if not dimension == self.dimension:
            raise
        expand_dims = np.ones((n,1))
        x = np.concatenate((expand_dims, x), axis=1) ## 增广化

        yhat = np.matmul(x, self.w)
        yhat = np.sign(yhat)
        correct = len(np.where(yhat == y)[0])

        print('accuracy %.2f'%(correct/n))

class DualSVM(SVM):
    def __init__(self, dimension=2):
        self.dimension = dimension
        self.w = np.zeros((dimension,1))  ## 不做增广化
        self.b = 0

    def quadratic_programming(self, z, y):
        '''
        Args:
            z: (n, dim) features after non-linear transform
            y: (n, 1) labels
            n: dimension after non-linear transform (not include constant 1)
        Description:
            solve convex optim problem
            min { 1/2 * u.T * Q * u + p.T * u }
            s.t: A * u <= C & R * u = v
            
            Dual SVM: the dimension of param are irrelated to ~d
            u: (n, 1)
            Q: (n, n)
            p: (n, 1)
            A: (n, n)
            C: (n, 1)
            R: (1, n)
            v: (1, 1)
        '''
        n, dimension = z.shape
        if not dimension == self.dimension:
            raise

        Q = np.matmul(y,y.T) * np.matmul(z, z.T)
        p = -1 * np.ones((n,1))
        A = -1 * np.eye(n)
        c = np.zeros((n,1))
        R = y.T
        v = np.zeros((1,1))

        Q = matrix(Q)
        p = matrix(p)
        A = matrix(A)
        c = matrix(c)
        R = matrix(R)
        v = matrix(v)

        alpha = np.array(qp(Q, p, A, c, R, v)['x']) ## (n, 1)
        idx = np.where(alpha>1e-8)[0][0]

        self.w = np.sum(alpha * y * z, axis=0).reshape(dimension, 1)
        self.b = y[idx] - np.matmul(z[idx], self.w)
        self.support_vector = np.where(alpha>1e-8)[0]

    def eval(self, x, y):
        n, dimension = x.shape
        if not dimension == self.dimension:
            raise

        yhat = np.matmul(x, self.w) + self.b
        yhat = np.sign(yhat)
        correct = len(np.where(yhat == y)[0])

        print('accuracy %.2f'%(correct/n))

class Kernel_SVM(SVM):
    def __init__(self, dimension=2):
        self.dimension = dimension

if __name__=='__main__':

    x = np.random.randn(100,2)
    y = np.concatenate((np.ones((50,1)),-1 * np.ones((50,1))), axis=0)
    model = SVM()
    epoch = 50
    lr = 0.1
    model.train(x, y, epoch=epoch, lr=lr)