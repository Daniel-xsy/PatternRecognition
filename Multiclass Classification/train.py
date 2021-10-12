from model import PLA, Softmax
from data import IrisDataset, MNIST
import numpy as np

def vote_ensemble_pred(model1, model2, model3, data):
    '''
    ovo vote ensemble
    '''
    data1 = np.array(data[0])
    data2 = np.array(data[1])
    data3 = np.array(data[2])
    x = np.concatenate([data1[:,:-1], data2[:,:-1], data3[:,:-1]],axis=0)
    y = np.concatenate([data1[:,-1], data2[:,-1], data3[:,-1]],axis=0)
    ## 三个二分类器分别计算
    y1 = model1.predict(x, label=[0,1])
    y2 = model2.predict(x, label=[0,2])
    y3 = model3.predict(x, label=[1,2])
    y_pred = np.concatenate((y1,y2,y3),axis=1)
    yhat = np.array([np.argmax(np.bincount(y_pred[k])) for k in range(y.shape[0])]).reshape(-1,1)
    
    return y.reshape(-1,1), yhat.reshape(-1,1)

def train_ovo_classifier(model, data1, data2):
    data1[:,4] = 1
    data2[:,4] = -1
    x = np.concatenate((data1[:,:-1],data2[:,:-1]),axis=0)
    y = np.concatenate((data1[:,-1],data2[:,-1]),axis=0)
    model.train(x,y)
    return model



def train_iris(mode='ovo',dataset=IrisDataset(),eval=True):
    '''
    Args:
        mode: ovo for one-verus-one classifier based on PLA
              softmax for softmax classifier
        dataset: instance of `class` IrisDataset
    '''
    if mode == 'ovo':
        ## 三个二分类模型
        classifier1 = PLA(dimension=4)
        classifier2 = PLA(dimension=4)
        classifier3 = PLA(dimension=4)

        train_data, test_data = dataset.split_data()
        ## 分别训练
        classifier1 = train_ovo_classifier(classifier1, train_data[0], train_data[1])
        classifier2 = train_ovo_classifier(classifier2, train_data[0], train_data[2])
        classifier3 = train_ovo_classifier(classifier3, train_data[1], train_data[2])
        
        if eval:
            y, yhat = vote_ensemble_pred(classifier1, classifier2, classifier3, test_data)
            correct = len(np.where(y==yhat)[0])
            print('accuracy: %.2f %%'%(correct*100/y.shape[0]))

        return classifier1, classifier2, classifier3

    if mode == 'softmax':
        train_x, test_x, train_y, test_y = dataset.get_data()
        train_x, test_x = train_x.T, test_x.T
        train_y = train_y.astype(int)
        test_y = test_y.astype(int)

        classifier = Softmax(cls_num=3, dimension=4)
        loss, acc = classifier.train(train_x, train_y, lr=0.1, epoch=1000, batchsize=60)
        
        return classifier, loss, acc
        

def train_mnist(root='./MNIST', lr=0.02, epoch=10, batchsize=256):
    classifier = Softmax()
    mnist = MNIST(root=root)

    x_train = mnist.x_train
    y_train = mnist.y_train
    x_test = mnist.x_test
    y_test = mnist.y_test
    a = 1


if __name__=='__main__':

    # root = 'iris.csv'
    # dataset = IrisDataset(root)
    # train_iris(mode='softmax', dataset=dataset)

    train_mnist()
