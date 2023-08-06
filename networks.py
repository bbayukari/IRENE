import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms


class SurvivalDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.img_dir = img_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.09, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.data = pd.read_csv(csv_file)
        self.data = self.data.loc[
            self.data["eid"].isin(
                [int(file.split(".")[0]) for file in os.listdir(img_dir)]
            )
        ]
        self.X = torch.from_numpy(self.data.iloc[:, 1:54].values).to(torch.float32)
        self.e = torch.from_numpy(self.data.iloc[:, 54].values).to(torch.float32)
        self.T = torch.from_numpy(self.data.iloc[:, 55].values).to(torch.float32)

    def __len__(self):
        return self.data.shape[0]
    
    def normalize(self):
        self.X = (self.X-self.X.min(axis=0)) / (self.X.max(axis=0)-self.X.min(axis=0))

    def __getitem__(self, index):
        img_item = self.transform(
            Image.open(self.img_dir + str(self.data.iloc[index, 0]) + ".jpg")
        )
        x_item = self.X[index]
        e_item = self.e[index]
        t_item = self.T[index]
        return img_item, x_item, e_item, t_item


class DeepSurv(nn.Module):
    """The module class performs building network according to config"""

    def __init__(self, drop, norm, dims, activation):
        super(DeepSurv, self).__init__()
        # parses parameters of network from configuration
        self.drop = drop
        self.norm = norm
        self.dims = dims
        self.activation = activation
        # builds network
        self.model = self._build_network()

    def _build_network(self):
        """Performs building networks according to parameters"""
        layers = []
        for i in range(len(self.dims) - 1):
            if i and self.drop is not None:  # adds dropout layer
                layers.append(nn.Dropout(self.drop))
            # adds linear layer
            layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if self.norm:  # adds batchnormalize layer
                layers.append(nn.BatchNorm1d(self.dims[i + 1]))
            # adds activation layer
            layers.append(eval("nn.{}()".format(self.activation)))
        # layers.append(nn.Sigmoid())
        # builds sequential network
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)


class Regularization(object):
    def __init__(self, order, weight_decay):
        """The initialization of Regularization class

        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        """
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        """Performs calculates regularization(self.order) loss for model.

        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        """
        reg_loss = 0
        for name, w in model.named_parameters():
            if "weight" in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss


class NegativeLogLikelihood(nn.Module):
    def __init__(self, l2_reg):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = l2_reg
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):
        mask = torch.ones(y.shape[0], y.shape[0]).to(torch.device("cuda"))
        y = y.unsqueeze(1)
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)
        l2_loss = self.reg(model)
        return neg_log_loss + l2_loss


class ImageCNN(nn.Module):
    def __init__(self, features, pool):
        super(ImageCNN, self).__init__()
        self.features = features
        self.pool = pool

    def forward(self, image, data):
        x = nn.Conv2d(1, self.features, (3, 3), padding="same")(image)
        x = nn.ReLU()(x)
        x = nn.BatchNorm2d(num_features=32, affine=False)(x)
        x = nn.MaxPool2d(kernel_size=self.pool)(x)
        x = torch.flatten(x)
        x = nn.ReLU()(x)
        return x
