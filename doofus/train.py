import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .callbacks import state_to_features, ACTIONS, n_features, Net, device
from typing import List
import numpy as np
import os

path=os.path.dirname(os.path.abspath(__file__))

def setup_training(self):

    self.optimizer = optim.SGD(self.Qnet.parameters(), lr=0.0001, momentum=0.8)
    #self.optimizer = optim.Adam(self.Qnet.parameters(), lr=0.00001)
    self.criterion = nn.SmoothL1Loss().to(device)

    self.memory = []

    self.R_tot  = 0.0
    self.R_tots = []

    self.Loss   = 0.0
    self.Losses = []

    self.update_counter = 0

    self.target_Qnet = Net().to(device)
    self.target_Qnet.double()
    self.target_Qnet.load_state_dict(self.Qnet.state_dict())
    self.target_Qnet.eval()

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state != None:

        self.update_counter += 1

        R = get_reward(self,old_game_state,new_game_state,events)
        #print(R, self_action)
        i = ACTIONS.index(self_action)
        self.R_tot += R

        #save experience in memory
        if len(self.memory)>500:
            self.memory.pop(0)

        if new_game_state != None:
            self.memory.append([torch.from_numpy(state_to_features(old_game_state)).to(device),
                                torch.from_numpy(state_to_features(new_game_state)).to(device),
                                i,
                                torch.tensor([R]).to(device).double(),
                                True,
                                10.0])
        else:
            self.memory.append([torch.from_numpy(state_to_features(old_game_state)).to(device),
                                torch.from_numpy(state_to_features(old_game_state)).to(device),
                                i,
                                torch.tensor([R]).to(device).double(),
                                False,
                                10.0])

        #train on mini-batch
        if len(self.memory)>1:

            priorities = np.array([s[5] for s in self.memory])
            importance = priorities/np.sum(priorities)
            k = np.random.choice(len(self.memory),size=np.clip(len(self.memory),None,5),replace=False,p=importance)

            mask_terminal_states = np.ones(len(k))
            for j,l in enumerate(k):
                if self.memory[l][4] == False: mask_terminal_states[j] = 0
            mask = torch.from_numpy(mask_terminal_states).double().to(device)

            batch_old = torch.stack([self.memory[k_][0] for k_ in k],0)
            batch_new = torch.stack([self.memory[k_][1] for k_ in k],0)
            batch_rew = torch.stack([self.memory[k_][3] for k_ in k])
            batch_a = torch.tensor([self.memory[k_][2] for k_ in k]).to(device)

            pred_value = self.Qnet(batch_old).gather(1,batch_a.view(-1,1))

            expected_value = batch_rew + (0.9 * mask * self.target_Qnet(batch_new).detach().max(1)[0]).unsqueeze(1)

            loss = self.criterion(pred_value,expected_value)
            self.Loss += loss.item()
            for k_ in k: self.memory[k_][5]=loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    else:
        pass


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    game_events_occurred(self, last_game_state, last_action, None, events)

    torch.save(self.Qnet.state_dict(),path+"\params.pt")

    if self.update_counter>100:
        self.target_Qnet.load_state_dict(self.Qnet.state_dict())
        self.target_Qnet.eval()
        self.update_counter = 0

    self.R_tots.append(self.R_tot/last_game_state["step"])
    np.save(path+"/reward.npy",self.R_tots)

    self.Losses.append(self.Loss/last_game_state["step"])
    np.save(path+"/loss.npy",self.Losses)

    self.R_tot=0.0
    self.Loss =0.0

def get_reward(self,old,new,events):

    R = 0

    if new != None:

        if "INVALID_ACTION" in events:
            R += -1

        tot_danger = np.sum(state_to_features(old)[9:18])
        if tot_danger<5 and ("WAITED" in events):
            R += -1

        if "COIN_COLLECTED" in events:
            R += 25
        elif len(old["coins"])>0 and len(new["coins"])>0 and old["self"][-2]:
            rc_old = np.sum(np.abs(state_to_features(old)[-4:-2]))
            rc_new = np.sum(np.abs(state_to_features(new)[-4:-2]))
            R += 10 * (rc_old - rc_new)

        if old["self"][-2] and len(old["coins"])==0:
            rb_old = np.sum(np.abs(state_to_features(old)[-2:]))
            rb_new = np.sum(np.abs(state_to_features(new)[-2:]))
            R += 5 *(rb_old - rb_new)

        if not 'BOMB_DROPPED' in events:
            danger = state_to_features(old)[13] - state_to_features(new)[13]
            R += 3 * danger
        else:
            if (state_to_features(old)[[1,3,5,7]] == 1).any() or (state_to_features(old)[:9] == 10).any():
                R += 10
            else:
                R += -50

    else:
        if "INVALID_ACTION" in events:
            R += -1

        tot_danger = np.sum(state_to_features(old)[9:18])
        if tot_danger < 5 and ("WAITED" in events):
            R += -1

        if "COIN_COLLECTED" in events:
            R += 25

        if 'KILLED_SELF' in events:
            R += -50

        if 'GOT_KILLED' in events:
            R += -50

    return R