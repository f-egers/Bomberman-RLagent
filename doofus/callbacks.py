import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
n_features = 22

device = "cpu"   #"cuda" or "cpu"

strategy = "softmax"   #"greedy" or "softmax" (avoids loops)

def setup(self):

    self.Qnet = Net().to(device)
    self.Qnet.double()

    if os.path.isfile("params.pt"):
        self.Qnet.load_state_dict(torch.load("params.pt"))


def act(self,game_state: dict):

    if game_state != None:

        if len(game_state["bombs"])>0:
            self.strategy = "greedy"
        else:
            self.strategy = strategy

        random_prob = np.clip(0.995**game_state["round"],0.1,None)
        if self.train and np.random.rand() < random_prob:
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

        self.Qnet.eval()
        self.model = self.Qnet(torch.from_numpy(state_to_features(game_state)).to(device))
        self.Qnet.train()
        #print(self.model)

        if self.strategy=="greedy":
            #print(ACTIONS[torch.argmax(self.model)])
            return ACTIONS[torch.argmax(self.model)]
        else:
            return np.random.choice(ACTIONS,p=torch.softmax(self.model/2,-1).cpu().detach().numpy().flatten())

    else:
        pass

def state_to_features(game_state: dict):

    x, y = game_state["self"][-1]

    field = game_state["field"]
    if game_state["self"][2]:
        field[x,y] = 1
    for o in game_state["others"]:
        field[o[-1]] = 10
    view = field[x-1:x+2,y-1:y+2].flatten()

    expl  = np.clip(np.array(game_state["explosion_map"])*100,0,10)
    for b in game_state["bombs"]:
        xb,yb = b[0]
        for i in range(1, 5):
            if field[xb + i, yb] == -1:
                break
            expl[xb + i, yb] = 10-1*b[1]-1*i
        for i in range(1, 5):
            if field[xb - i, yb] == -1:
                break
            expl[xb - i, yb] = 10-1*b[1]-1*i
        for i in range(1, 5):
            if field[xb, yb + i] == -1:
                break
            expl[xb, yb + i] = 10-1*b[1]-1*i
        for i in range(1, 5):
            if field[xb, yb - i] == -1:
                break
            expl[xb, yb - i] = 10-1*b[1]-1*i
        expl[xb,yb] = 10-1*b[1]

    danger_lev = expl[x-1:x+2,y-1:y+2].flatten()

    if len(game_state["coins"])>0:
        coins = np.array(game_state["coins"])
        dist_coins = np.sum(np.abs(coins - np.array([x,y])[None,:]),axis=-1)
        coin = coins[np.argmin(dist_coins)].flatten()-np.array([x,y])
    else:
        coin = np.array([0,0])

    box_ind = np.where(field == 1)
    if len(box_ind[0])>0:
        boxes = np.array(list(zip(box_ind[0],box_ind[1])))
        dist_boxes = np.sum(np.abs(boxes - np.array([x, y])[None, :]), axis=-1)
        box = boxes[np.argmin(dist_boxes)].flatten() - np.array([x, y])
    else:
        box = np.array([0,0])

    return np.concatenate((view,danger_lev,coin,box))




#network class
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(n_features, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, len(ACTIONS)))

    def forward(self, x):
        return self.fc(x)

