from sklearn.utils import shuffle
from data import Iris,MNIST
from model import NeuralNetwork, LeNet
from torch.utils.data import DataLoader,Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utlis import init_weight


def train_iris(model:nn.Module, dataset:Dataset, optimizer:optim.Optimizer, device,
                    epoch, batchsize, eval=True, log=True):

    dataloder = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    x_test, y_test = dataset.x_test, dataset.y_test

    acc_test = []
    acc_train = []
    losses = []

    for n_epoch in range(epoch):

        correct_train = torch.zeros(1).squeeze()
        loss_per_epoch = torch.zeros(1).squeeze()

        for i,(x,y) in enumerate(dataloder):
            x = x.float().to(device)
            y = y.squeeze().long().to(device)
            yhat = model(x)
            loss_train = F.cross_entropy(yhat, y)
            loss_per_epoch += loss_train.cpu()
            loss_train.backward()
            optimizer.step()

            prediction = torch.argmax(yhat, dim=-1)
            correct_train += (prediction == y).sum().float().cpu()
            
        loss_per_epoch = loss_per_epoch/ (i + 1)
        accuracy_train = correct_train / len(dataset)

        losses.append(loss_per_epoch)
        acc_train.append(accuracy_train)

        if eval:
            with torch.no_grad():
                x = torch.from_numpy(x_test).float().to(device)
                y = torch.from_numpy(y_test).long().squeeze().to(device)

                yhat = model(x)
                prediction = torch.argmax(yhat, dim=-1)
                correct_test = (prediction == y).sum().float().cpu()
                
                accuracy_test = correct_test / len(y)
                acc_test.append(accuracy_test)

        if log:
            print('epoch: %i/%i\t training loss %.2f\t training_acc %.2f\t val_acc %.2f'%
                        (n_epoch+1, epoch, loss_per_epoch, accuracy_train, accuracy_test))
        
    return model, losses, acc_train, acc_test
            


def train_mnist(model:nn.Module, dataset:Dataset, optimizer:optim.Optimizer, device,
                    epoch, batchsize, norm=True, bacthnorm=True, eval=True, log=True):

    if norm:
        dataset.normalize()

    dataloder = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    x_test, y_test = dataset.x_test, dataset.y_test

    acc_test = []
    acc_train = []
    losses = []

    for n_epoch in range(epoch):

        correct_train = torch.zeros(1).squeeze()
        loss_per_epoch = torch.zeros(1).squeeze()

        for i,(x,y) in enumerate(dataloder):
            x = x.float().view(-1, 1, 28, 28).to(device)
            y = y.squeeze().long().to(device)
            yhat = model(x, batch_norm=bacthnorm)
            loss_train = F.cross_entropy(yhat, y)
            loss_per_epoch += loss_train.cpu()
            loss_train.backward()
            optimizer.step()

            prediction = torch.argmax(yhat, dim=-1)
            correct_train += (prediction == y).sum().float().cpu()
            
        loss_per_epoch = loss_per_epoch/ (i + 1)
        accuracy_train = correct_train / len(dataset)

        losses.append(loss_per_epoch)
        acc_train.append(accuracy_train)

        if eval:
            with torch.no_grad():
                x = torch.from_numpy(x_test).float().view(-1, 1, 28, 28).to(device)
                y = torch.from_numpy(y_test).long().squeeze().to(device)

                yhat = model(x, batch_norm=bacthnorm)
                prediction = torch.argmax(yhat, dim=-1)
                correct_test = (prediction == y).sum().float().cpu()
                
                accuracy_test = correct_test / len(y)
                acc_test.append(accuracy_test)

        if log:
            print('epoch: %i/%i\t training loss %.2f\t training_acc %.2f\t val_acc %.2f'%
                        (n_epoch+1, epoch, loss_per_epoch, accuracy_train, accuracy_test))
    
    return model, losses, acc_train, acc_test


def main_trainer(mode='iris'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    if mode == 'iris':
        root = './iris.csv'

        layers = [4,10,10,3]
        activation = 'sigmoid'
        epoch = 100
        lr = 0.1
        momentum = 0
        batchsize = 90
        weight_decay = 0.1

        model = NeuralNetwork(layers=layers, activation=activation)
        model = model.to(device)
        model.apply(init_weight)
        iris_dataset = Iris(root=root, split_ratio=0.4)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        model, _, _, _ = train_iris(model, iris_dataset, optimizer, device, epoch, batchsize)
        print(model)

    elif mode == 'mnist':
        root = './MNIST'

        epoch = 10
        lr = 0.001
        momentum = 0
        batchsize = 256
        weight_decay = 0.1
        norm = True
        batchnorm = True

        model = LeNet()
        model = model.to(device)
        model.apply(init_weight)
        mnist_dataset = MNIST(root=root)
        #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
        model, _, _, _ = train_mnist(model, mnist_dataset, optimizer, device, epoch, batchsize, norm, batchnorm)
        print(model)

    else:
        raise


if __name__=='__main__':
    mode = ['iris','mnist']
    main_trainer(mode[1])

    