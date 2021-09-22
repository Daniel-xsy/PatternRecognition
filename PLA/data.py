import numpy as np

class PointData(object):
    def __init__(self, train_num=8, test_num=2, linear=True):
        super(PointData,self).__init__()
        self.linear = linear
        self._generate_data(train_num, test_num)

    def _generate_data(self, train_num, test_num):
        total_num = train_num + test_num
        if self.linear:
            means1 = np.array([3,6])
            means2 = np.array([6,3])
            covar = np.array([1.5,1,1,1.5]).reshape(2,2)
        else:
            means1 = np.array([4,5])
            means2 = np.array([5,4])
            covar = np.array([1.5,1,1,1.5]).reshape(2,2)
        x1 = np.random.multivariate_normal(means1, covar, size=total_num)
        y1 = np.ones((total_num, 1))
        x2 = np.random.multivariate_normal(means2, covar, size=total_num)
        y2 = np.ones((total_num, 1)) * -1

        self.x_train = np.concatenate((x1[:train_num], x2[:train_num]))
        self.y_train = np.concatenate((y1[:train_num], y2[:train_num]))
        self.x_test = np.concatenate((x1[train_num:], x2[train_num:]))
        self.y_test = np.concatenate((y1[train_num:], y2[train_num:]))


if __name__=='__main__':
    dataset = PointData()
    print(dataset.x_train)