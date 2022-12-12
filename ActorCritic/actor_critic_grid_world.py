from ast import main
import collections
from statistics import mean
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import matplotlib.lines as mlines
from math import cos, pi, sin, sqrt
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_size, 16)  # 5*5 from image dimension
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # x = self.fc1(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class Value(nn.Module):
    def __init__(self, input_size, output_size):
        super(Value, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_size, 16)  # 5*5 from image dimension
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.softmax(x)
        return x


class ActorCritic:
    def __init__(self, gamma, num_rows=5, num_cols=5):
        self.actions = [(-1,0), (0,1), (1,0), (0,-1)] # up, right, down, left 
        self.arrows = ["↑", "→","↓", "←"]
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.gamma = gamma


        self.pi_star = [[(0, 1), (0, 1), (0, 1), (1, 0), (1, 0)],
                   [(0, 1), (0, 1), (0, 1), (1, 0), (1, 0)],
                   [(-1, 0), (-1, 0), (-1, 0), (1, 0), (1, 0)],
                   [(-1, 0), (-1, 0), (-1, 0), (1, 0), (1, 0)],
                   [(-1, 0), (-1, 0), (0, 1), (0, 1), (-1, 0)]]
        self.v_star = [[4.0187,4.5548,5.1576,5.8337,6.4553],
                       [4.3716,5.0324,5.8013,6.6473,7.3907],
                       [3.8672,4.39,  0.0,   7.5769,8.4637],
                       [3.4183,3.8319,0.0,   8.5738,9.6946],
                       [2.9978,2.9309,6.0733,9.6946,0.]]


        self.states = [(i, j) for j in range(self.num_cols) for i in range(self.num_rows)]
        self.v = np.array([[0.0 for j in range(self.num_cols)] for i in range(self.num_rows)])
        # self.test_pol = np.array([[[1/len(self.actions) for k in range(len(self.actions))] for j in range(self.num_cols)] for i in range(self.num_rows)])
        self.q = np.array([[[0.0 for a in range(len(self.actions))] for j in range(self.num_cols)] for i in range(self.num_rows)])
        # self.w1 = torch.zeros((self.num_rows*self.num_cols*len(self.actions), 1), requires_grad=True, dtype=torch.float32)
        # self.b1 = torch.zeros((1, 1), requires_grad=True, dtype=torch.float32)
        self.test_pol = Policy(self.num_rows*self.num_cols, len(self.actions))
        self.v = Value(self.num_rows*self.num_cols, 1)

    def trans_func(self, s, a):
        """
        Args:
            s = tuple(row, col)
            a = action (tuple)
        Returns:
            next state (tuple)
        """
        if s == (4, 4): return s
        rand = random.uniform(0, 1)

        if rand < 0.8:
            s_prime = (s[0] + a[0], s[1] + a[1])
        elif 0.8 < rand <  0.85:
            a = self.actions[(self.actions.index(a) + 1) % 4]
            s_prime = (s[0] + a[0], s[1] + a[1])
        elif  0.85 < rand <  0.9:
            a = self.actions[(self.actions.index(a) - 1) % 4]
            s_prime = (s[0] + a[0], s[1] + a[1])
        else:
            s_prime = s
        if (s_prime == (2,2)) or (s_prime == (3,2)) or (s_prime[0] < 0) or (s_prime[0] > 4) or (s_prime[1] < 0) or (s_prime[1] > 4):
            s_prime = s
        return s_prime

    def reward(self, s, a, s_prime):
        if (s == (4, 4)):
            return 0.0
        elif s_prime == (4, 4):
            return 10.0
        elif s_prime == (4, 2):
            return -10.0
        else:
            return 0.0

    def d0(self):
        states = self.states.copy()
        states.remove((2,2))
        states.remove((3,2))
        states.remove((4,4))
        random_index = random.randint(0, len(states) - 1)
        return states[random_index]


    def generateEpisode(self, pi):
        trajectory = []
        s = self.d0()
        while(s != (4, 4)):
            a = self.pi_func(pi, s)
            s_prime = self.trans_func(s, a)
            r = self.reward(s, a, s_prime)
            trajectory.append((s, r))
            s = s_prime
        trajectory.append(((4,4), 0))
        return trajectory


    def pi_func(self, pi, s, epsilon):
        """
        Args: 
            pi: dictionary key = states, value = list of actions
            s = tuple(row, col)
            eps = float
        Returns:
            action in state s (tuple)
        """
        # probs = np.array([])
        # for i, expl_a in enumerate(self.actions):
        #     probs = np.append(probs, (self.get_q_hat(s, expl_a)).tolist()[0][0])
        x = self.get_x(s)
        out = self.test_pol(x)
        probs = out.detach().numpy()
        chosen_action = np.random.choice(len(self.actions),1,p=probs)

        # probs = out.tolist()
        # probs = [round(p, 4) for p in probs]
        # probs = np.exp(probs)
        # probs /= sum(probs)
        # chosen_action = np.random.choice(4, 1, p=probs)
        return self.actions[chosen_action[0]], chosen_action, out
        a_star = list(np.argwhere(probs == max(probs)).flatten())
        prob = np.random.uniform()
        if prob < epsilon:
            return self.actions[random.sample([0, 1, 2, 3], 1)[0]], probs
        else:
            return self.actions[random.sample(a_star, 1)[0]], probs


    def get_x(self, state):
        x = torch.zeros(self.num_cols*self.num_rows)
        index = state[0]*self.num_cols + state[1]
        if state not in set([(3,2), (2,2), (4,4)]):
            x[index] = 1
        return x

    def get_v(self, s):
        if s in set([(3,2), (2,2), (4,4)]):
            return torch.zeros(1, requires_grad=True)
        return self.v(self.get_x(s))
    
    # def get_q_hat(self, s, a):
    #     if s in set([(3,2), (2,2), (4,4)]):
    #         return torch.zeros(1,1)
    #     x = self.get_x(s, a)
    #     ans = torch.matmul(x, self.w1) + self.b1
    #     return ans

    # def calculate_reward(self, tau, n, T, r_trajectory):
    #     result = 0
    #     for i in range(tau+1, min(tau+n, T) + 1):
    #         result += (self.gamma**(i-tau-1)) * r_trajectory[i]
    #     return result


    def actor_critic(self, alpha_theta, alpha_w, eps, epochs):
        count = 0
        optimizer_theta = torch.optim.Adam(self.test_pol.parameters(), lr=alpha_theta)
        optimizer_w = torch.optim.Adam(self.v.parameters(), lr=alpha_w)
        max_norm = []
        scheduler_theta = CosineAnnealingWarmRestarts(optimizer_theta, T_0=10, eta_min=1e-7)
        scheduler_w = CosineAnnealingWarmRestarts(optimizer_w, T_0=10, eta_min=1e-7)
        while True:
            print(count)
            count += 1
            I = 1
            s = self.d0()

            # eps = max(0.01, eps*(0.99**count))
            # alpha = max(0.001, alpha*(0.99**count))
            while s != (4, 4):

                a, a_index, out = self.pi_func(self.test_pol, s, eps)
                s_prime = self.trans_func(s, a)
                r = self.reward(s, a, s_prime)
                # if r!= 0: print(r, s_prime)
                # print(torch.argmax(self.get_x(s)), s[0], s[1])
                v_s = self.get_v(s)
                # print(v_s)
                v_s_prime = self.get_v(s_prime)
                delta = r + self.gamma * v_s_prime - v_s
                # print(delta)
                # print("a_index = {}".format(a_index))
                policy_loss = (alpha_theta * I * delta * torch.log(out.squeeze(0)[a_index]))
                v_loss = (alpha_w * delta * v_s)


                policy_loss.backward(retain_graph=True)
                v_loss.backward(retain_graph=True)
                with torch.no_grad(): 
                    optimizer_theta.step()
                    optimizer_w.step()
                optimizer_theta.zero_grad()
                optimizer_w.zero_grad()

                # out_policy.backward()
                # v_s.backward()
                # # print((alpha_w * delta).tolist())
                # # optimizer = torch.optim.Adam(self.v.parameters(), lr=-(alpha_w*delta).tolist()[0])
                # with torch.no_grad():  
                #     for param_w in self.v.parameters():
                #         # print("param_w = {}".format(param_w))
                #         grad = param_w.grad
                #         # print(torch.sum(grad), torch.argmax(self.get_x(s)))
                #         # print(torch.sum(alpha_w * delta * grad))
                #         param_w += alpha_w * delta * grad
                #         # print("param_w = {}".format(torch.sum(param_w)))
                #         param_w.grad.zero_()


                #     for param_theta in self.test_pol.parameters():
                #         param_theta += alpha_theta * I * delta * param_theta.grad
                #         param_theta.grad.zero_()
                # optimizer.step()
                I = self.gamma * I
                s = s_prime
                # print(s_prime)
            scheduler_theta.step()
            scheduler_w.step()
            v_test = self.get_v_final()
            max_norm.append(np.amax(np.abs(v_test - self.v_star)))

            if count > epochs:
                break

        return count, max_norm

    def get_v_final(self):
        v_val = np.array([[0.0 for j in range(self.num_cols)] for i in range(self.num_rows)])
        for i in range(5):
            for j in range(5):
                if (i == 2 and j == 2) or (i == 3 and j == 2) or (i == 4 and j == 4):
                    v_val[i, j] = 0.0    
                else:
                    v_val[i, j] = self.v(self.get_x((i,j)))
        return v_val

    def get_policy(self):
        k = 0
        for i in range(5):
            for j in range(5):
                probs = np.array([])
                for a in self.actions:
                    probs = np.append(probs, (self.get_q_hat((i,j), a)).tolist()[0][0])
                if i == 4 and j == 4:
                    print("G", end = " ")
                elif (i == 2 or i == 3) and j == 2:
                    print(" ", end = " ")
                else:
                    probs = np.exp(probs)
                    probs /= sum(probs)
                    ind = np.argmax(probs)
                    print(self.arrows[ind], end=" ")
                    # print(probs, end=" ")
                    k += 1
            print()

    def mse(self, m1, m2):
        return np.square(np.subtract(m1, m2)).mean() 

def main():
    gamma = 0.9
    n = 8
    alpha_theta = 1e-5
    alpha_w = 1e-5
    eps = 0.05
    epochs = 10000
    actor_critic = ActorCritic(gamma)
    count, max_norm = actor_critic.actor_critic(alpha_theta, alpha_w, eps, epochs)
    plt.plot(max_norm)
    plt.show()
    v_test = actor_critic.get_v_final()
    print(v_test)
    
    # episodic.get_policy()
    
    # print(v_test)
    # print("MSE = {}".format(np.square(np.subtract(v_test, episodic.v_star)).mean()))
    # print("max_norm = {}".format(np.amax(np.abs(v_test - episodic.v_star))))
    
    # plt.figure(0)
    # plt.plot(num_actions_list)
    # plt.xlabel("Number of episode")
    # plt.ylabel("Number of actions")

    # plt.figure(1)
    # plt.plot(max_norm)
    # plt.xlabel("Number of episodes")
    # plt.ylabel("Max Norm")

    # plt.figure(2)
    # plt.plot(mse_list)
    # plt.xlabel("Number of episodes")
    # plt.ylabel("MSE")

    # plt.show()
if __name__ == '__main__':
    main()
