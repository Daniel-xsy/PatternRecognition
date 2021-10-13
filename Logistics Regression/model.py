import numpy as np
        
class LogisticRegression:
    def __init__(self, lr=0.02, epoch=50):
        '''
        mode: g: Grandient Descent
              a: analysis
        '''
        self.w = np.zeros((3,1))
        self.lr = lr
        self.epoch = epoch

    def __call__(self, x, y):

        x = x.reshape(-1,2)
        y = y.reshape(-1,1)
        expand_axis = np.ones((x.shape[0],1))
        x = np.concatenate((expand_axis, x), axis=-1)
        ## print(x.shape)
    

        losses = []
        for i in range(self.epoch):
            yhat = self.sigmoid(np.matmul(x, self.w))
            loss = self.cross_entropy(x, y, yhat)
            losses.append(loss)
            grad = self._calculate_grad(x, y, yhat)
        
            self.w -= self.lr * grad
            print('epoch: %i/%i loss: %.2f'%(i+1, self.epoch, loss))

        return yhat, losses

    def _calculate_grad(self, x, y, yhat):
        batchsize = y.shape[0]
        s = np.matmul(x, self.w)
        grad = (self.sigmoid(-y*s) - 1) * y * x
        grad = np.sum(grad, axis=0) / batchsize
        return grad.reshape(3,1)

    def cross_entropy(self,x, y, yhat):
        num = y.shape[0]
        s = np.matmul(x, self.w)
        loss = (1/num)*np.sum(np.log(1+np.exp(-s*y)))
        return loss

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def eval(self, x, y):
        test_num = x.shape[0]
        expand_axis = np.ones((x.shape[0],1))
        x = np.concatenate((expand_axis, x), axis=-1)
        yhat = np.matmul(x, self.w)
        yhat = np.sign(self.sigmoid(yhat) - 0.5)
        assert(len(yhat) == len(y))
        correct_num = len(np.where(yhat == y)[0])
        print('accuracy: %.2f'%(correct_num/test_num))