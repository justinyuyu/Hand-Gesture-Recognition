import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import cv2
import os

# # CNN Model


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn1 = nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(8, 32, kernel_size=2, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc = nn.Linear(10368, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)

        out = self.fc(out)
        return out

################ grasp five image ################


def prediction(img_lst, model):
    device = torch.device('cpu')
    model = CNN()
    #model = torch.load("weight/no_skeleton.pkl")
    model.load_state_dict(torch.load('skeleton_demo.pkl'))
    model.to(device)

    # store predict result
    res = []
    model.eval()
    for img in img_lst:

        #img = cv2.imread("", cv2.IMREAD_COLOR)

        dim = (150, 150)
        # resize image to 150*150
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = np.transpose(img, (2, 0, 1))

        img_tensor = torch.from_numpy(
            np.array([img])).to(device, dtype=torch.float)
        label_tensor = torch.from_numpy(np.array([0])).type(torch.LongTensor)
        test_set = TensorDataset(img_tensor, label_tensor)
        test_loader = DataLoader(
            test_set, batch_size=1, shuffle=False, num_workers=0)

        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x = x.to(device, dtype=torch.float)
                output = model(x)
                pred = output.argmax(dim=1)

                res.append(int(pred[0]))

    vals, counts = np.unique(np.array(res), return_counts=True)
    index = np.argmax(counts)
    predict_number = vals[index]
    return predict_number