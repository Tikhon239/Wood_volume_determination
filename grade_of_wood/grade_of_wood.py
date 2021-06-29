import os
import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from seaborn import heatmap
import matplotlib.pyplot as plt

from config import Config
from models.pretrained_model import PretrainedModel
from dataset.grade_of_wood_dataset import MyDataset, MySimpleDataset, balance_dataset_split
from metrics.metrics import get_accuracy, get_accuracy_classes, get_f1_score, get_f1_score_classes, get_confusion_matrix


def train(net, train_dataloader, val_dataloader, save_path, epochs=100, lr=0.01):
    device = next(net.parameters()).device
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    train_loss_array = []
    val_loss_array = []
    for epoch in tqdm(range(1, epochs + 1), total=epochs):
        average_train_loss = 0
        net.train()
        for i, l in train_dataloader:
            i = i.to(device)
            l = l.to(device)
            optimizer.zero_grad()
            pl = net(i)
            loss = F.cross_entropy(pl, l.long())
            loss.backward()
            optimizer.step()
            average_train_loss += loss.item()
        average_train_loss /= len(train_dataloader)

        average_val_loss = 0
        net.eval()
        for i, l in val_dataloader:
            i = i.to(device)
            l = l.to(device)
            with torch.no_grad():
                pl = net(i)
            loss = F.cross_entropy(pl, l.long())
            average_val_loss += loss.item()
        average_val_loss /= len(train_dataloader)

        train_loss_array.append(average_train_loss)
        val_loss_array.append(average_val_loss)
        print(f"epoch : {epoch}, avarege train loss {average_train_loss}, avarege val loss {average_val_loss}")
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_array,
                'val_loss': val_loss_array
            }, save_path + f"_{epoch}.pth")
    return train_loss_array, val_loss_array


if __name__ == "__main__":
    config = Config()
    path = "data_4"
    data_name = "csv/table_nek_2020_01.csv"
    dataset = MySimpleDataset(path, data_name)
    classes = dataset.label_encoder.inverse_transform(np.arange(4))

    train_set, val_set = balance_dataset_split(dataset, 0.5)
    val_set, test_set = balance_dataset_split(val_set, 0.5)

    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=1, drop_last=False)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=1, drop_last=False)
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=1, drop_last=False)

    model = PretrainedModel(output_size=4, freeze_model=False, model_name="resnet").to(config.device)
    save_path = os.path.join("weights", "wood_model_resnet")
    train_loss_array, val_loss_array = train(model, train_dataloader, val_dataloader, save_path)

    print("Train accuracy:", get_accuracy(model, train_dataloader))
    print("Validation accuracy:", get_accuracy(model, val_dataloader))

    print('Train')
    get_accuracy_classes(model, train_dataloader, classes)
    print('Validation')
    get_accuracy_classes(model, val_dataloader, classes)

    plt.plot(np.log(train_loss_array))
    plt.plot(np.log(val_loss_array))
    plt.show()

    confusion_matrix = get_confusion_matrix(model, val_dataloader, classes)
    heatmap(confusion_matrix, cmap="YlGnBu")
    plt.show()
