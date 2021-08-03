# original code: https://github.com/jmichaux/dqn-pytorch/blob/master/models.py
import math
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from dqn import DQN, Memory
from environment import LumberJackEnv


class RLMgr:
    def __init__(self, env, n_episode=200, batch_size=32, eps_decay=0.005, lr=1e-4):
        self.env = env
        self.n_action = env.n_action
        self.n_episode = n_episode
        self.batch_size = batch_size
        self.memory = Memory(batch_size * 20)
        self.eps_decay = eps_decay
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = DQN(self.n_action).to(self.device)
        self.t_model = DQN(self.n_action).to(self.device)
        self.update_target_model()
        self.t_model.eval()
        
        if True:
            pass
        else:
            raise Exception('wrong model')
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def update_target_model(self):
        self.t_model.load_state_dict(self.model.state_dict())

    def run(self):
        step = 0
        for ep in range(self.n_episode):
            cur_img = self.env.reset()
            total_reward = 0
            while True:
                action = self.get_action(cur_img, step)
                next_img, reward, done = env.step(action)

                self.memory.push(cur_img, action, next_img, reward)

                cur_img = next_img
                step += 1
                total_reward += reward
                if step >= self.batch_size * 5:
                    self.train_model(step)
                    if step % (self.batch_size * 5) == 0:
                        self.update_target_model()
                        print('Update Target Model')
                if done:
                    break
            if ep % 10 == 0:
                print(f'[{ep+1}/{self.n_episode}] reward: {total_reward:.2f}, step: {step}')

    def get_action(self, cur_img, step):
        eps = 1 / math.exp(step * self.eps_decay)
        print(eps)
        if random.random() > eps:
            with torch.no_grad():
                cur_img = torch.FloatTensor(cur_img).to(self.device).unsqueeze(0).unsqueeze(0) / 255
                self.model.eval()
                pred = self.model(cur_img)
                action = pred.argmax(-1).item()
                return action
        else:
            action = random.randint(0, self.n_action - 1)  # [0, n_action)
            return action

    def train_model(self, step, gamma=0.99):
        self.model.train()
        batch = self.memory.sample(self.batch_size)

        cur_imgs = torch.FloatTensor(np.array(batch[:, 0].tolist())) / 255
        cur_imgs = cur_imgs.unsqueeze(1).to(self.device)

        next_imgs = torch.FloatTensor(np.array(batch[:, 2].tolist())) / 255
        next_imgs = next_imgs.unsqueeze(1).to(self.device)

        actions = torch.LongTensor(batch[:, 1].astype(np.int)).to(self.device).unsqueeze(1)
        rewards = torch.LongTensor(batch[:, 3].astype(np.int)).to(self.device)

        pred_rewards = self.model(cur_imgs).gather(1, actions)

        with torch.no_grad():
            gt_rewards = self.t_model(next_imgs).max(1)[0].detach()
        gt_rewards = gt_rewards * gamma + rewards
        gt_rewards = gt_rewards.unsqueeze(1)

        loss = F.smooth_l1_loss(pred_rewards, gt_rewards)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)  # gradient clipping
        self.optimizer.step()
        if step % (self.batch_size * 5) == 0:
            print('loss:', loss.item())


if __name__ == '__main__':
    TELGRAM_GAME_URL = 'https://tbot.xyz/lumber/'
    env = LumberJackEnv(TELGRAM_GAME_URL)
    rlmgr = RLMgr(env)
    rlmgr.run()

