import numpy as np

from sklearn.metrics import f1_score

import torch


def get_accuracy(net, dataloader):
    net.eval()
    device = next(net.parameters()).device

    total = 0
    correct = 0
    for X, y in dataloader:
        with torch.no_grad():
            y_hat = net(X.to(device))

        y_hat = torch.argmax(y_hat.cpu(), 1)
        total += len(y)
        correct += torch.eq(y, y_hat).sum().item()
    return 100 * correct / total


def get_accuracy_classes(net, dataloader, classes):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    net.eval()
    device = next(net.parameters()).device
    for X, y in dataloader:
        with torch.no_grad():
            y_hat = net(X.to(device))

        y_hat = torch.argmax(y_hat.cpu(), 1)
        for y_i, y_hat_i in zip(y, y_hat):
            correct_pred[classes[y_i]] += (y_i == y_hat_i)
            total_pred[classes[y_i]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * correct_count / total_pred[classname]
        print(f"Accuracy for class {classname} is: {accuracy}")


def get_confusion_matrix(net, dataloader, classes):
    confusion_matrix = np.zeros((len(classes), len(classes)))

    net.eval()
    device = next(net.parameters()).device
    for X, y in dataloader:
        with torch.no_grad():
            y_hat = net(X.to(device))

        y_hat = torch.argmax(y_hat.cpu(), 1)
        for y_i, y_hat_i in zip(y, y_hat):
            confusion_matrix[y_i, y_hat_i] += 1

    return confusion_matrix


def get_f1_score(net, dataloader):
    net.eval()
    device = next(net.parameters()).device

    avg_f1_score = 0
    for X, y in dataloader:
        with torch.no_grad():
            y_hat = net(X.to(device))
        _, y_hat = torch.max(y_hat.cpu(), 1)
        avg_f1_score += f1_score(y, y_hat, average='micro')
    return avg_f1_score / len(dataloader)


def get_f1_score_classes(net, dataloader, classes):
    classes_dict = {class_name: 0 for class_name in classes}

    net.eval()
    device = next(net.parameters()).device
    for X, y in dataloader:
        with torch.no_grad():
            y_hat = net(X.to(device))

        _, y_hat = torch.max(y_hat.cpu(), 1)
        for y_i, f_score in enumerate(f1_score(y, y_hat, average=None)):
            classes_dict[classes[y_i]] += f_score

    for class_name, f_score in classes_dict.items():
        avg_f_score = f_score / len(dataloader)
        print(f"F1 score for class {class_name} is: {avg_f_score}")