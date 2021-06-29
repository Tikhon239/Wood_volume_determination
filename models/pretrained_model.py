from torch import nn
import torchvision.models as models


class PretrainedModel(nn.Module):
    def __init__(self,
                 output_size: int,
                 freeze_model: bool = True,
                 model_name: str = "resnet"):
        super().__init__()
        self.model = None
        if model_name == "resnet":
            """ Resnet18"""
            self.model = models.resnet18(pretrained=True)
            self.set_parameter_requires_grad(self.model, freeze_model)

            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, output_size)

        elif model_name == "alexnet":
            """ Alexnet"""
            self.model = models.alexnet(pretrained=True)
            self.set_parameter_requires_grad(self.model, freeze_model)

            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, output_size)

        elif model_name == "vgg":
            """ VGG11_bn"""
            self.model = models.vgg11_bn(pretrained=True)
            self.set_parameter_requires_grad(self.model, freeze_model)

            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, output_size)

        elif model_name == "densenet":
            """ Densenet"""
            self.model = models.densenet121(pretrained=True)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, output_size)

    @staticmethod
    def set_parameter_requires_grad(model, freeze_model: bool):
        if freeze_model:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)
