import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):
    """
    Class for Convolutional Neural Network
    """
    def __init__(self, num_classes, img_rows, img_cols, img_channels=1):
        super().__init__()
        self.num_classes = num_classes
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.max_pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(in_channels=self.img_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # this line must be updated if the order in the forward function changes
        self.flattened_size = (128*self.img_rows*self.img_cols)//16 #*0.25*0.25

        self.hidden1 = nn.Linear(self.flattened_size, 128)
        self.hidden2 = nn.Linear(128, self.num_classes)


    def forward(self, x):
        x_out = self.conv1(x) # out is of size [bach, 32, img_rows, img_cols]
        x_out = self.relu(x_out)

        x_out = self.conv2(x_out)
        x_out = self.relu(x_out)
        x_out = self.max_pool(x_out)  # out is of size [bach, 64, img_rows*0.5, img_cols*0.5]

        x_out = self.conv3(x_out)
        x_out = self.relu(x_out)
        x_out = self.max_pool(x_out)  # out is of size [bach, 128, img_rows*0.25, img_cols*0.25]

        x_flat = self.flatten(x_out) # x_flat is of size [batch, 128*img_rows*img_cols*0.25*0.25]
        x_hidden = self.hidden1(x_flat)
        x_hidden = self.dropout(x_hidden)

        x_logis = self.hidden2(x_hidden)
        return x_logis