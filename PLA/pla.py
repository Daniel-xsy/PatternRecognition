import numpy as np

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
        print('W:')
        print(self.W)
        print('b')
        print(self.b)

    def train(self, x, y):
        num, dim = x.shape
        if dim != self.dimension:
            raise
        optimized = False
        while not optimized:
            for i in range(num):
                yhat = np.dot(self.W,x[i]) + self.b
                if yhat * y[i] <= 0:
                    self._update_param(x[i],y[i])
                    break
                # all data classify correctly
                if i == num-1:
                    optimized = True
        print('over training!')

    def inference(self, x, y):
        num, dim = x.shape
        if dim != self.dimension:
            raise
        yhat = np.matmul(self.W,x.transpose(1,0)) + self.b
        yhat = np.sign(yhat).transpose(1,0)
        accuracy = 1 - len(np.nonzero(yhat - y)[0])/len(y)
        print('accuracy: %.2f'%accuracy)

        return yhat

class Pocket(object):
    def __init__(self, dimension):
        super(Pocket,self).__init__()
        # zero initialization
        self.dimension = dimension
        self.W = np.zeros((1,dimension))
        self.b = 0

    def _error_eval(self, w, b, x, y):
        yhat = np.matmul(w, x.transpose(1,0)) + b
        yhat = np.sign(yhat).transpose(1,0)
        error_idxs = np.nonzero(yhat - y)[0]

        return len(error_idxs), error_idxs

    def train(self, x, y, show=False):
        # init pocket vector
        # chocie = np.random.randint(0,x.shape[0])
        # w = x[chocie].reshape(1,2)
        # b = y[chocie]
        w = np.random.randn(1,2)
        b = np.random.randn(1)
        # w = self.W
        # b = self.b
        print('w:')
        print(w)
        print('b:')
        print(b)
        for i in range(20):
            error_num, error_idxs = self._error_eval(w, b, x, y)
            if error_num == 0:
                break
            else:
                error_idx = np.random.choice(error_idxs)
                # updata w, b
                error_x = x[error_idx]
                error_y = y[error_idx]
                w_new = w + error_y * error_x
                b_new = b + error_y
                error_num_new, error_idxs_new = self._error_eval(w_new, b_new, x, y)
                if error_num_new < error_num:
                    # replace w, b with the better one
                    error_num = error_num_new
                    error_idxs = error_idxs_new
                    w = w_new
                    b = b_new
                    if show:
                        print('w:')
                        print(w)
                        print('b:')
                        print(b)
                        print('error: %i'%error_num)
        self.W = w
        self.b = b

    def inference(self, x, y):
        num, dim = x.shape
        if dim != self.dimension:
            raise
        yhat = np.matmul(self.W,x.transpose(1,0)) + self.b
        yhat = np.sign(yhat).transpose(1,0)
        accuracy = 1 - len(np.nonzero(yhat - y)[0])/len(y)
        print('accuracy: %.2f'%accuracy)

        return yhat





                
            




