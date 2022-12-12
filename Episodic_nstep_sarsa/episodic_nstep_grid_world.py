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

class EpisodciNstep:
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
        self.test_pol = np.array([[[1/len(self.actions) for k in range(len(self.actions))] for j in range(self.num_cols)] for i in range(self.num_rows)])
        self.q = np.array([[[0.0 for a in range(len(self.actions))] for j in range(self.num_cols)] for i in range(self.num_rows)])
        self.w1 = torch.zeros((self.num_rows*self.num_cols*len(self.actions), 1), requires_grad=True, dtype=torch.float32)
        self.b1 = torch.zeros((1, 1), requires_grad=True, dtype=torch.float32)

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
        probs = np.array([])
        for i, expl_a in enumerate(self.actions):
            probs = np.append(probs, (self.get_q_hat(s, expl_a)).tolist()[0][0])
        a_star = list(np.argwhere(probs == max(probs)).flatten())
        prob = np.random.uniform()
        if prob < epsilon:
            return self.actions[random.sample([0, 1, 2, 3], 1)[0]]
        else:
            return self.actions[random.sample(a_star, 1)[0]]


    def get_x(self, state, action):
        x = torch.zeros(100)
        a_index = self.actions.index(action)
        index = state[0]*20 + state[1]*4 + a_index
        if state != (4, 4):
            x[index] = 1
        return x


    def get_q_hat(self, s, a):
        if s in set([(3,2), (2,2), (4,4)]):
            return torch.zeros(1,1)
        x = self.get_x(s, a)
        ans = torch.matmul(x, self.w1) + self.b1
        return ans

    def calculate_reward(self, tau, n, T, r_trajectory):
        result = 0
        for i in range(tau+1, min(tau+n, T) + 1):
            result += (self.gamma**(i-tau-1)) * r_trajectory[i]
        return result


    def episodic_n_step(self, n, alpha, eps):
        count = 0
        max_norm = []
        mse_list = []
        mse_itr_number = []
        num_actions_list = []
        while True:
            s_trajectory = []
            a_trajectory = []
            r_trajectory = []
            count += 1
            T = float("inf")
            t = 0
            num_actions = 0
            s = self.d0()
            s_trajectory.append(s)
            # eps = max(0.01, eps*(0.99**count))
            # alpha = max(0.001, alpha*(0.99**count))
            a = self.pi_func(self.test_pol, s, eps)
            a_trajectory.append(a)
            r_trajectory.append(0.0)
            while True:
                num_actions += 1
                if t < T:

                    s_prime = self.trans_func(s, a)
                    r = self.reward(s, a, s_prime)
                    s_trajectory.append(s_prime)
                    r_trajectory.append(r)

                    if s_prime == (4, 4):
                        T = t + 1
                    else:
                        a_prime = self.pi_func(self.test_pol, s_prime, eps)
                        a_trajectory.append(a_prime)
                        a = a_prime
                
                s = s_prime
                tau = t - n + 1

                if tau >= 0:
                    G = self.calculate_reward(tau, n, T, r_trajectory)

                    if tau + n < T:
                        G += (self.gamma**n) * self.get_q_hat(s_trajectory[tau+n], a_trajectory[tau+n]).tolist()[0][0]

                    w1_d = torch.zeros(self.w1.size())
                    b1_d = torch.zeros(self.b1.size())
                    out = self.get_q_hat(s_trajectory[tau], a_trajectory[tau])
                    out.backward()
                    with torch.no_grad():

                        w1_grad = self.w1.grad
                        b1_grad = self.b1.grad
                        w1_d = (alpha * (G - out).tolist()[0][0] * w1_grad)
                        b1_d = (alpha * (G - out).tolist()[0][0] * b1_grad)
                        self.w1 += w1_d
                        self.b1 += b1_d
                        self.w1.grad.zero_()
                        self.b1.grad.zero_()
                if tau == T - 1:
                    break
                t += 1
            v_est = self.get_v()
            max_norm.append(np.amax(np.abs(v_est - self.v_star)))
            mse_list.append(self.mse(v_est, self.v_star))
            num_actions_list.append(num_actions)
            
            # if count % 1 == 0:
            #     mse_list.append(self.mse(v_est, self.v_star))
            #     mse_itr_number.append(count)
            # print(count)
            if count > 1000:
                break
        return count, num_actions_list, max_norm, mse_list

    def get_v(self):
        for i in range(5):
            for j in range(5):
                q_val = np.array([])
                for a in self.actions:
                    q_val = np.append(q_val, (self.get_q_hat((i,j), a)).tolist()[0][0])
                self.v[i, j] = max(q_val)
        return self.v

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
    alpha = 0.1
    eps = 0.05
    episodic = EpisodciNstep(gamma)
    count, num_actions_list, max_norm, mse_list = episodic.episodic_n_step(n, alpha, eps)

    episodic.get_policy()
    v_test = episodic.get_v()
    
    print(v_test)
    print("MSE = {}".format(np.square(np.subtract(v_test, episodic.v_star)).mean()))
    print("max_norm = {}".format(np.amax(np.abs(v_test - episodic.v_star))))
    
    plt.figure(0)
    plt.plot(num_actions_list)
    plt.xlabel("Number of episode")
    plt.ylabel("Number of actions")

    plt.figure(1)
    plt.plot(max_norm)
    plt.xlabel("Number of episodes")
    plt.ylabel("Max Norm")

    plt.figure(2)
    plt.plot(mse_list)
    plt.xlabel("Number of episodes")
    plt.ylabel("MSE")

    plt.show()
if __name__ == '__main__':
    main()
