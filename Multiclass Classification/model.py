import numpy as np
import math

class PLA(object):
    def __init__(self, dimension):
        super(PLA,self).__init__()
        # zero initialization
        self.dimension = dimension
        self.W = np.zeros((1,dimension))
        self.b = 0
    
    def _update_param(self, x, y):
        self.W += x * y
        self.b += y

    def train(self, x, y):
        num, dim = x.shape
        if dim != self.dimension:
            raise
        optimized = False
        count = 0
        while not optimized:
            for i in range(num):
                if count == 1000:
                    optimized = True
                    break
                yhat = np.dot(self.W,x[i]) + self.b
                if yhat * y[i] <= 0:
                    self._update_param(x[i],y[i])
                    count += 1 
                    break
                # all data classify correctly
                if i == num-1:
                    optimized = True
        print('over training!')

    def predict(self, x, label=[0,1]):
        num, dim = x.shape
        if dim != self.dimension:
            raise
        yhat = np.matmul(self.W,x.transpose(1,0)) + self.b
        yhat = (np.sign(yhat).transpose(1,0))
        result = np.ones((num,1))
        result[np.where(yhat==1)[0]] = label[0]
        result[np.where(yhat==-1)[0]] = label[1]

        return result.astype(int)

    def eval(self, x, y):
        num, dim = x.shape
        if dim != self.dimension:
            raise
        yhat = np.matmul(self.W,x.transpose(1,0)) + self.b
        yhat = np.sign(yhat).transpose(1,0)
        accuracy = 1 - len(np.nonzero(yhat - y)[0])/len(y)
        print('accuracy: %.2f'%accuracy)

        return yhat

class Softmax:
    def __init__(self, cls_num=10, dimension=784):
        self.cls_num = cls_num
        self.dimension = dimension
        self.w = np.random.normal(0, 0.1, (cls_num,dimension+1)) ##增广化

    def train(self, x, y, lr=0.1, epoch=10, batchsize=256):
        dimension, train_num = x.shape
        if not dimension == self.dimension:
            raise
        expand_dim = np.ones((1,train_num))
        x = np.concatenate((expand_dim,x),axis=0)

        iteration = math.ceil(train_num / batchsize)
        losses = []
        acc = []
        for n_epoch in range(epoch):
            for n_iteration in range(iteration):
                start = n_iteration * batchsize
                end = min((n_iteration + 1) * batchsize, train_num)
                x_iter = x[:,start:end]
                y_iter = y[start:end]

                yhat = self.softmax(np.matmul(self.w,x_iter))
                loss = self.cross_entropy(y_iter, yhat)
                losses.append(loss)

                grad = self._calculate_grad(x_iter, yhat, y_iter)
                self.w -= lr*grad

                yhat = np.argmax(yhat,axis=0).reshape(-1,1)
                correct = len(np.where(y_iter==yhat)[0])
                acc_temp = correct/(end-start)
                acc.append(acc_temp)

                print('epoch %i/%i iteration %i/%i loss %.2f acc %.2f'%
                                            (n_epoch,epoch,n_iteration,iteration,loss,acc_temp))

        return loss, acc

    def predict(self, x):
        dimension, train_num = x.shape
        if not dimension == self.dimension:
            raise
        expand_dim = np.ones((1,train_num))
        x = np.concatenate((expand_dim,x),axis=0)

        yhat = self.softmax(np.matmul(self.w,x))
        yhat = np.argmax(yhat,axis=0)
        return yhat

    def _calculate_grad(self, x, yhat, y):
        dimension, batchsize = x.shape
        temp = yhat.copy()
        for i in range(batchsize):
            temp[y[i],i] -= 1
        grad = np.matmul(temp, x.T)/batchsize

        return grad

    def cross_entropy(self, y, yhat):
        loss = 0
        batchsize = y.shape[0]
        seq = np.arange(batchsize)
        b = yhat.T[seq,y.T]
        a = -np.log(yhat.T[seq,y.T])
        loss = np.sum(-np.log(yhat.T[seq,y.T]),axis=1)/batchsize

        return loss


    def softmax(self, x):
        exp = np.exp(x)
        exp_sum = np.sum(exp, axis=0)
        x = exp / exp_sum
        return x


if __name__=='__main__':
    model = Softmax()
    x = np.random.randn(784,1000)
    y = np.random.randint(low=0,high=10,size=(1000,1))
    lr = 0.1
    epoch = 15
    batchsize = 128
    model.train(x,y,lr,epoch,batchsize)