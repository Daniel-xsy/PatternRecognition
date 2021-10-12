import csv
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST
import struct


class IrisDataset:
    def __init__(self,root='iris.csv'):
        self.root = root
        self.name = ['setosa','versicolor','virginica']
        self.data = self._load_data()
        a = 1

    def _load_data(self):
        if not os.path.isfile(self.root):
            raise()

        data = {}
        for i in range(len(self.name)):
            data[self.name[i]] = []
        with open(self.root,'r') as f:
            reader = csv.reader(f)
            reader.__next__()
            for row in reader:
                data[row[5]].append(list(map(float,row[1:-1])))

        return data

    ## used for training ovo classifier
    def split_data(self, ratio=0.2):
        '''
        Args:
            ratio: test_num / total_num
        '''
        train_data = []
        test_data = []
        for i in range(len(self.name)):
            data = self.data[self.name[i]]
            num_cls = len(data)
            num_test = int(num_cls * ratio)

            train_data.append(np.array(data[:-num_test]))
            test_data.append(np.array(data[-num_test:]))

            label = np.ones((num_cls-num_test,1)) * i
            train_data[i] = np.concatenate((train_data[i],label), axis=1)
            label = np.ones((num_test,1)) * i
            test_data[i] = np.concatenate((test_data[i],label), axis=1)

        return train_data, test_data

    ## used for training softmax classifier
    def get_data(self, ratio=0.2):
        label_num = [len(self.data[self.name[i]]) for i in range(len(self.name))]
        label = [i * np.ones((label_num[i],1)) for i in range(len(label_num))]
        label = np.concatenate(label)
        x = [self.data[self.name[i]] for i in range(len(self.name))]
        x = np.concatenate(x)

        train_x, test_x, train_y, test_y = train_test_split(x, label, test_size=ratio)

        return train_x, test_x, train_y, test_y
        
class MNIST(object):
    '''
    MNIST数据集类
    '''
    def __init__(self, root='./MNIST'):
        '''
        方法说明:
            初始化类
        参数说明:
            root: 文件夹根目录
            image_file: mnist图像文件 'train-images.idx3-ubyte' 'test-images.idx3-ubyte'
            label_file: mnist标签文件 'train-labels.idx1-ubyte' 'test-labels.idx1-ubyte'
        '''
        img_train_path = os.path.join(root, 'train-images.idx3-ubyte')
        label_train_path = os.path.join(root, 'train-labels.idx1-ubyte')
        img_test_path = os.path.join(root, 'test-images.idx3-ubyte')
        label_test_path = os.path.join(root, 'test-labels.idx1-ubyte')
        
        self.x_train = self._get_img(img_train_path)
        self.y_train = self._get_label(label_train_path)
        self.x_test = self._get_img(img_test_path)
        self.y_test = self._get_label(label_test_path)
        self.standardlize()

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
        
        return images.T

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
    def normalize(self):
        
        min = np.min(self.x_train, axis=0).reshape(1,-1)
        max = np.max(self.x_train, axis=0).reshape(1,-1)
        self.x_train = (self.x_train - min)/(max - min)

        min = np.min(self.x_test, axis=0).reshape(1,-1)
        max = np.max(self.x_test, axis=0).reshape(1,-1)
        self.x_test = (self.x_test - min)/(max - min)

    #数据归一化
    def standardlize(self):
        
        mean = np.mean(self.x_train, axis=0).reshape(1,-1)
        var = np.var(self.x_train, axis=0).reshape(1,-1)
        self.x_train = (self.x_train-mean)/np.sqrt(var)

        mean = np.mean(self.x_test, axis=0).reshape(1,-1)
        var = np.var(self.x_test, axis=0).reshape(1,-1)
        self.x_test = (self.x_test-mean)/np.sqrt(var)
        

if __name__=='__main__':
    dataset = MNIST()