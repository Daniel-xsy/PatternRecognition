from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import csv
import struct

class Iris(Dataset):
    def __init__(self, root='./iris.csv', split_ratio=0.4):
        self.name = {'setosa':0, 'versicolor':1, 'virginica':2}
        super(Iris,self).__init__()
        self.root = root
        self.x_train, self.y_train, \
                self.x_test, self.y_test = self._load_data(ratio=split_ratio)

    def _load_data(self, ratio=0.4):
        if not os.path.isfile(self.root):
            raise FileNotFoundError

        x = []
        y = []
        with open(self.root,'r') as f:
            reader = csv.reader(f)
            ## 跳过首行
            reader.__next__()
            for row in reader:
                x.append(list(map(float,row[1:-1])))
                y.append(float(self.name[row[-1]]))

        x = np.array(x).reshape(150,4)
        y = np.array(y).reshape(150,1)

        x1_train, x1_test, y1_train, y1_test = train_test_split(x[:50], y[:50], test_size=ratio)
        x2_train, x2_test, y2_train, y2_test = train_test_split(x[50:100], y[50:100], test_size=ratio)
        x3_train, x3_test, y3_train, y3_test = train_test_split(x[100:150], y[100:150], test_size=ratio)

        x_train = np.concatenate((x1_train,x2_train,x3_train), axis=0)
        y_train = np.concatenate((y1_train,y2_train,y3_train), axis=0)
        x_test = np.concatenate((x1_test,x2_test,x3_test), axis=0)
        y_test = np.concatenate((y1_test,y2_test,y3_test), axis=0)

        return x_train, y_train.astype(int), x_test, y_test.astype(int)

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, index: int):
        return self.x_train[index], self.y_train[index]

class MNIST(Dataset):
    def __init__(self, root='./MNIST'):
        '''
        方法说明:
            初始化类
        参数说明:
            root: 文件夹根目录
            image_file: mnist图像文件 'train-images.idx3-ubyte' 'test-images.idx3-ubyte'
            label_file: mnist标签文件 'train-labels.idx1-ubyte' 'test-labels.idx1-ubyte'
        '''
        super(Dataset,self).__init__()

        img_train_path = os.path.join(root, 'train-images.idx3-ubyte')
        label_train_path = os.path.join(root, 'train-labels.idx1-ubyte')
        img_test_path = os.path.join(root, 'test-images.idx3-ubyte')
        label_test_path = os.path.join(root, 'test-labels.idx1-ubyte')
        
        self.x_train = self._get_img(img_train_path)
        self.y_train = self._get_label(label_train_path)
        self.x_test = self._get_img(img_test_path)
        self.y_test = self._get_label(label_test_path)

    #读取图片
    def _get_img(self, path):

        with open(path,'rb') as fi:
            ImgFile = fi.read()
            head = struct.unpack_from('>IIII', ImgFile, 0)
            #定位数据开始位置
            offset = struct.calcsize('>IIII')
            ImgNum = head[1]
            width = head[2]
            height = head[3]
            #每张图片包含的像素点
            pixel = height*width
            bits = ImgNum * width * height
            bitsString = '>' + str(bits) + 'B'
            #读取文件信息
            images = struct.unpack_from(bitsString, ImgFile, offset)
            #转化为n*726矩阵
            images = np.reshape(images,[ImgNum,pixel])
        
        return images

    #读取标签
    def _get_label(self, path):

        with open(path,'rb') as fl:
            LableFile = fl.read()
            head = struct.unpack_from('>II', LableFile, 0)
            labelNum = head[1]
            #定位标签开始位置
            offset = struct.calcsize('>II')
            numString = '>' + str(labelNum) + "B"
            labels = struct.unpack_from(numString, LableFile, offset)
            #转化为1*n矩阵
            labels = np.reshape(labels, [labelNum])

        return labels

    #数据标准化
    def normalize(self, epsilon=1e-6):
        
        min = np.min(self.x_train, axis=0).reshape(1,-1)
        max = np.max(self.x_train, axis=0).reshape(1,-1)
        self.x_train = (self.x_train - min)/(max - min + epsilon)

        min = np.min(self.x_test, axis=0).reshape(1,-1)
        max = np.max(self.x_test, axis=0).reshape(1,-1)
        self.x_test = (self.x_test - min)/(max - min + epsilon)

    #数据归一化
    def standardlize(self, epsilon=1e-6):
        
        mean = np.mean(self.x_train, axis=0).reshape(1,-1)
        var = np.var(self.x_train, axis=0).reshape(1,-1)
        self.x_train = (self.x_train-mean)/(np.sqrt(var) + epsilon)

        mean = np.mean(self.x_test, axis=0).reshape(1,-1)
        var = np.var(self.x_test, axis=0).reshape(1,-1)
        self.x_test = (self.x_test-mean)/(np.sqrt(var) + epsilon)

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, index: int):
        return self.x_train[index], self.y_train[index]

if __name__=='__main__':
    # dataset = Iris()
    dataset = MNIST()