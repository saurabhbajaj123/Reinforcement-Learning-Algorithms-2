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
    def __init__(self, gamma, num_rows, num_cols):
        self.actions = [(-1,0), (0,1), (1,0), (0,-1)] # up, right, down, left 
        self.arrows = ["↑", "→","↓", "←"]
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.gamma = gamma


        self.states = [(i, j) for j in range(self.num_cols) for i in range(self.num_rows)]
        self.cliff_states = [(3, j) for j in range(1, 11)]
        self.goal_state = (3, 11)
        self.start_state = (3, 0)
        self.v = np.array([[0.0 for j in range(self.num_cols)] for i in range(self.num_rows)])
        # self.policy = np.array([[(0, 1) for j in range(self.num_cols)] for i in range(self.num_rows)])
        self.test_pol = np.array([[[1/len(self.actions) for k in range(len(self.actions))] for j in range(self.num_cols)] for i in range(self.num_rows)])
        self.q = np.array([[[0.0 for a in range(len(self.actions))] for j in range(self.num_cols)] for i in range(self.num_rows)])
        # self.test_policy = nn.Linear(self.num_rows*self.num_cols, len(self.actions))
        # self.v = nn.Linear(self.num_rows*self.num_cols, 1)
        # self.q = nn.Linear(self.num_rows*self.num_cols*len(self.actions), 1)

        self.w1 = torch.zeros((self.num_rows*self.num_cols*len(self.actions), 1), requires_grad=True, dtype=torch.float32)
        self.b1 = torch.zeros((1, 1), requires_grad=True, dtype=torch.float32)
        
        # torch.nn.init.normal_(self.w1, mean=0.0, std=1.0)
        # torch.nn.init.normal_(self.b1, mean=0.0, std=1.0)

    def trans_func(self, s, a):
        """
        Args:
            s = tuple(row, col)
            a = action (tuple)
        Returns:
            next state (tuple)
        """
        if s == self.goal_state: return s
        s_prime = (s[0] + a[0], s[1] + a[1])
        if (s_prime[0] < 0) or (s_prime[0] > 3) or (s_prime[1] < 0) or (s_prime[1] > 11):
            s_prime = s
        if s_prime in self.cliff_states:
            s_prime = self.start_state
        return s_prime

    def reward(self, s, a, s_prime):
        if s_prime in self.cliff_states:
            return -100
        else:
            return -1

    def d0(self):
        states = self.states.copy()
        for j in range(1, 11):
            states.remove((3, j))
        random_index = random.randint(0,len(states)-1)
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
        # print("probabilities = {}".format(pi[s[0]][s[1]]))
        probs = np.array([])
        for i, expl_a in enumerate(self.actions):
            probs = np.append(probs, (self.get_q_hat(s, expl_a)).tolist()[0][0])
        # print((probs))
        a_star = list(np.argwhere(probs == max(probs)).flatten())
        # print(a_star)
        prob = np.random.uniform()
        # print(random.sample([0, 1, 2, 3], 1)[0])
        # print(random.sample(a_star, 1)[0])
        if prob < epsilon:
            return self.actions[random.sample([0, 1, 2, 3], 1)[0]]
        else:
            return self.actions[random.sample(a_star, 1)[0]]
        # print("probs = {}".format(probs))
        probs = np.exp(probs)
        probs /= sum(probs)
        chosen_action = np.random.choice(4, 1, p=probs)
        # print(chosen_action)
        return self.actions[chosen_action[0]]

    def e_soft_policy_update(self, s, eps):
        row = s[0]
        col = s[1]

        best_a_list = []
        best_qsa = -float("inf")
        
        for i, expl_a in enumerate(self.actions):
            if best_qsa < self.q[row][col][i]:
                best_qsa = self.q[row][col][i]
                best_a_list = [i]
            elif best_qsa == self.q[row][col][i]:
                best_a_list.append(i)

        not_best_list = list(set(range(4)) - set(best_a_list))
        new_prob = max(0, ((1- eps)/len(best_a_list)) + (eps/len(self.actions)))
        remaining_prob = (eps/len(self.actions))
        np.put(self.test_pol[row][col], best_a_list, [new_prob]*len(best_a_list))
        np.put(self.test_pol[row][col], not_best_list, [remaining_prob]*len(not_best_list))

    def softmax_policy_update(self, s, sigma):
        p = sigma*self.q[s[0]][s[1]]
        self.test_pol[s[0]][s[1]] = np.exp(p - max(p))/sum(np.exp(p - max(p)))

    def get_x(self, s, a):
        a_index = self.actions.index(a)
        x = torch.zeros(self.num_rows * self.num_cols * len(self.actions))
        # x = np.zeros((self.num_rows * self.num_cols * len(self.actions)))
        index = s[0]*(self.num_cols * len(self.actions)) + s[1]*len(self.actions) + a_index
        if s != self.goal_state:
            x[index] = 1
        return x

    def get_q_hat(self, s, a):
        if s == self.goal_state or s in self.cliff_states:
            return torch.zeros((1, 1), requires_grad=True, dtype=torch.float32)
        x = self.get_x(s, a)
        # print(x)
        ans = torch.matmul(x, self.w1) + self.b1
        # print(ans.size())
        # print(ans.tolist())
        return ans

    def calculate_reward(self, tau, n, T, r_trajectory):
        result = 0
        for i in range(tau+1, min(tau+n, T) + 1):
            result += (self.gamma**(i-tau-1)) * r_trajectory[i]
            # if result > 0.0:
            #     # print(result)
        # print(result)
        return result

    def print_policy(self):
        print("Policy:") # Printing policy
        k = 0
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if (i, j) == self.goal_state:
                    print("G", end = " ")
                elif (i, j) in self.cliff_states:
                    print(" ", end = " ")
                else:
                    ind = np.argmax(self.test_pol[i][j])
                    print(self.arrows[ind], end=" ")
                    k += 1
            print()

    def episodic_n_step(self, n, alpha, eps):
        count = 0
        max_norm = []
        while True:
            s_trajectory = []
            a_trajectory = []
            r_trajectory = []
            count += 1
            T = float("inf")
            t = 0
            s = self.d0()
            s_trajectory.append(s)
            a = self.pi_func(self.test_pol, s, eps)
            a_trajectory.append(a)
            r_trajectory.append(0.0)
            while True:
                if t < T:

                    s_prime = self.trans_func(s, a)
                    r = self.reward(s, a, s_prime)
                    s_trajectory.append(s_prime)
                    r_trajectory.append(r)

                    if s_prime == self.goal_state:
                        T = t + 1
                    else:
                        a_prime = self.pi_func(self.test_pol, s_prime, eps)
                        a_trajectory.append(a_prime)
                        a = a_prime
                
                s = s_prime
                tau = t - n + 1
                # print("tau = {}".format(tau))

                if tau >= 0:
                    # if s_trajectory[tau+n] in set([(3,2), (2,2), (4,4)]):
                    #     continue
                    G = self.calculate_reward(tau, n, T, r_trajectory)

                    if tau + n < T:
                        # print(type(self.get_q_hat(s_trajectory[tau+n], a_trajectory[tau+n]).tolist()[0][0]))
                        G += (self.gamma**n) * self.get_q_hat(s_trajectory[tau+n], a_trajectory[tau+n]).tolist()[0][0]
                    # if G > 0.0:
                    #     print(G)
                    # print("self.w1 = {}".format(self.w1))
                    w1_d = torch.zeros(self.w1.size())
                    b1_d = torch.zeros(self.b1.size())
                    # with torch.no_grad():
                    # print("s_tau = {}, s_tau+n = {}".format(s_trajectory[tau], s_trajectory[tau+n]))
                    out = self.get_q_hat(s_trajectory[tau], a_trajectory[tau])
                    out.backward()
                    with torch.no_grad():

                        w1_grad = self.w1.grad
                        b1_grad = self.b1.grad
                        # print("w1_grad = {}, b1 = {}".format(w1_grad, b1_grad))
                        # print((G - out).tolist()[0][0])
                        w1_d = (alpha * (G - out).tolist()[0][0] * w1_grad)
                        b1_d = (alpha * (G - out).tolist()[0][0] * b1_grad)
                        self.w1 += w1_d
                        self.b1 += b1_d
                        # print(self.w1, self.b1)
                        self.w1.grad.zero_()
                        self.b1.grad.zero_()
                if tau == T - 1:
                    break

                t += 1
            if count > 5000:
                break
        return count 

def main():
    gamma = 0.9
    n = 8
    alpha = 0.1
    eps = 0.05
    rows = 4
    cols = 12
 
    cliff_states = []
    start_state = (3, 0)
    goal_state = (3, 11)

    for j in range(1, 11):
        cliff_states.append((3, j))

    episodic = EpisodciNstep(gamma, rows, cols)
    episodic.episodic_n_step(n, alpha, eps)
    # for s in episodic.states:
    k = 0
    for i in range(rows):
        for j in range(cols):
            probs = np.array([])
            for a in episodic.actions:
                probs = np.append(probs, (episodic.get_q_hat((i,j), a)).tolist()[0][0])
            if (i, j) == goal_state:
                print("G", end = " ")
            elif (i, j) in cliff_states:
                print(" ", end = " ")
            else:
                ind = np.argmax(probs)
                print(episodic.arrows[ind], end=" ")
                k += 1

        print()

    for i in range(rows):
        for j in range(cols):
            q_val = np.array([])
            for a in episodic.actions:
                q_val = np.append(q_val, (episodic.get_q_hat((i,j), a)).tolist()[0][0])
            episodic.v[i, j] = max(q_val)
    
    print(episodic.v)
    # print("MSE = {}".format(np.square(np.subtract(episodic.v, episodic.v_star)).mean()))
    # print("max_norm = {}".format(np.amax(np.abs(episodic.v - episodic.v_star))))
    

if __name__ == '__main__':
    main()
