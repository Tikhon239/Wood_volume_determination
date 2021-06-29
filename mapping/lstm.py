import os
import numpy as np
from time import time
from tqdm.notebook import tqdm

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from config import Config
from models.lstm_model import Model, calculate_loss
from dataset.lstm_dataset import MyDataset, MySimpleDataset, collate_fn, balance_dataset_split
from metrics.mapping_metrics import get_rnn_accuracy


def train(net, train_dataloader, val_dataloader, save_path, epochs=100, lr=0.01, weights=None):
    device = next(net.parameters()).device
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    train_loss_array = []
    val_loss_array = []
    for epoch in tqdm(range(1, epochs + 1), total=epochs):
        average_train_loss = 0
        net.train()
        for i, l, s in train_dataloader:
            i = i.to(device)
            l = l.to(device)

            optimizer.zero_grad()
            pl = net(i, s)
            loss = calculate_loss(pl, l, weights)
            loss.backward()
            optimizer.step()
            average_train_loss += loss.item()
        average_train_loss /= len(train_dataloader)

        average_val_loss = 0
        net.eval()
        for i, l, s in val_dataloader:
            i = i.to(device)
            l = l.to(device)
            with torch.no_grad():
                pl = net(i, s)
            loss = calculate_loss(pl, l, weights)
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
    # dataset = MyDataset(path = "/content/drive/MyDrive/data",
    #                     data_name = "/content/drive/MyDrive/csv/table_nek_2020_01.csv",
    #                     down_scale = 8,
    #                     config = config,
    #                     number_of_strips = 5)

    path = "data_4"
    data_name = "csv/table_nek_2020_01.csv"
    dataset = MySimpleDataset(path, data_name, number_of_strips=5)

    train_set, validation_set = balance_dataset_split(dataset, 0.5)
    val_set, test_set = balance_dataset_split(validation_set, 0.5)

    train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=1,
                                  drop_last=False)
    val_dataloader = DataLoader(val_set, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=1,
                                drop_last=False)
    test_dataloader = DataLoader(test_set, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=1,
                                 drop_last=False)

    # weights = torch.zeros(3, dtype=torch.float32).to(config.device)
    # for path_folder in dataset.path_folders:
    #     l = dataset.get_labels(path_folder)
    #     for i in range(3):
    #         weights[i] += (l == i).sum()
    #
    # w = 1 / weights
    # w /= w.min()

    model = Model(input_size=128, output_size=3, hidden_size=128, num_layers=2, freeze_model=False, device=config.device)
    model.to(config.device)

    save_path = os.path.join("weights", "rnn_model_resnet")
    train_loss_array, val_loss_array = train(model, train_dataloader, val_dataloader, save_path, epochs=50)

    plt.plot(np.log(train_loss_array))
    plt.plot(np.log(val_loss_array))
    plt.show()

    images, labels = dataset[0]
    start_time = time()
    with torch.no_grad():
        mapping_result = np.argmax(model(images.to(config.device)).detach().to('cpu'), axis=2)
    print("Mapping time:", time()-start_time)

    print('Global accuracy', get_rnn_accuracy(model, validation_set))
