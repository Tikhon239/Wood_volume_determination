import torch
from torch import nn
import torch.nn.functional as F

from models.pretrained_model import PretrainedModel


class LSTMModel(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 num_layers: int,
                 device: str = "cpu"):
        super().__init__()
        self.device = device

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.end_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        h0, c0 = self.init_hidden(batch_size)
        out, _ = self.lstm(x, (h0, c0))

        out = self.end_fc(out)
        return out

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return h, c


class Model(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 freeze_model: bool = True,
                 model_name: str = "resnet",
                 device: str = "cpu"):
        super().__init__()
        self.input_size = input_size
        self.device = device
        self.encoder = PretrainedModel(input_size, freeze_model, model_name)

        self.lstm = LSTMModel(input_size, output_size, hidden_size, num_layers, device)

    def forward(self, x, seq_sizes=None, ):
        z = self.encoder(x)
        if seq_sizes is None:
            embeddings_batch = z[None, :].to(self.device)
        else:
            embeddings = torch.split(z, seq_sizes)

            embeddings_batch = torch.full(
                (len(seq_sizes), max(seq_sizes), self.input_size), 0, dtype=torch.float32
            ).to(self.device)

            for i, embedding in enumerate(embeddings):
                embeddings_batch[i, :embedding.shape[0]] = embedding

        return self.lstm(embeddings_batch)


def calculate_loss(predicted_labels: torch.Tensor, labels: torch.Tensor, weight=None) -> torch.Tensor:
    """
    Calculate cross entropy with ignoring PAD index
    :param predicted_labels: [batch size; seq length, vocab size]
    :param labels: [batch size; seq length]
    :return: [1]
    """
    batch_size = labels.shape[0]
    # [batch size; vocab size; seq length]
    predicted_labels = predicted_labels.permute(0, 2, 1)

    # [batch size; seq length]
    if weight is None:
        loss = F.cross_entropy(predicted_labels, labels.long(), ignore_index=-1, reduction="none")
    else:
        loss = F.cross_entropy(predicted_labels, labels.long(), weight=weight, ignore_index=-1, reduction="none")
    loss = loss.sum() / batch_size
    return loss
