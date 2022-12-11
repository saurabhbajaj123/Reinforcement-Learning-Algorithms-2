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
from queue import PriorityQueue


class PrioritizedSweeping:
    def __init__(self, alpha, gamma, delta, num_bins):
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta
        self.num_rows = num_bins + 1
        self.num_cols = num_bins + 1

        self.bins_x = np.linspace(start=-1.2, stop=0.5, num=num_bins)
        self.bins_v = np.linspace(start=-0.07, stop=0.07, num=num_bins)

        self.states = [(i, j) for j in range(self.num_cols) for i in range(self.num_rows)]
        self.actions = [-1, 1]
        self.v = np.array([[0.0 for j in range(self.num_cols)] for i in range(self.num_rows)])
        self.q = np.array([[[0.0 for a in range(len(self.actions))] for j in range(self.num_cols)] for i in range(self.num_rows)])
        self.test_pol = np.array([[[float(1/len(self.actions)) for k in range(len(self.actions))]for j in range(self.num_cols)] for i in range(self.num_rows)])

        self.model = np.array([[[(0.0, (0, 0)) for a in range(len(self.actions))] for j in range(self.num_cols)] for i in range(self.num_rows)]) # this is a 3D matrix, with each value being a pair (reward, state)
        self.pq = PriorityQueue()
        self.predecessor = defaultdict(list, {k : [] for k in self.states})


    def trans_func(self, s, a):
        x_t, v_t = s
        v_t_prime = v_t + 0.001*a - 0.0025*cos(3*x_t)
        x_t_prime = x_t + v_t_prime

        if x_t_prime < -1.2:
            x_t_prime = -1.2
        elif x_t_prime > 0.5:
            x_t_prime = 0.5
        
        if v_t_prime < -0.07:
            v_t_prime = -0.07
        elif v_t_prime > 0.07:
            v_t_prime = 0.07
        
        if x_t_prime == -1.2 or x_t_prime == 0.5:
            v_t_prime = 0

        # x_t_prime = np.digitize([x_t_prime], self.bins)[0]
        # v_t_prime = np.digitize([v_t_prime], self.bins)[0]
        return (x_t_prime, v_t_prime)
    
    def pi_func(self, pi, s):
        row = np.digitize([s[0]], self.bins_x)[0]
        col = np.digitize([s[1]], self.bins_v)[0]
        chosen_action = np.random.choice(len(self.actions), 1, p=pi[row][col])
        return self.actions[chosen_action[0]]

    def d0(self):
        s_0 = random.uniform(-0.6, -0.4)
        # s_0 = np.random.normal(loc=-0.3, scale=0.1)
        return (s_0, 0)

    def reward_function(self, state):
        if state[0] == 0.5:
            return 0
        else:
            return -1
    
    def estimate_J(self, N):
        G = 0
        T = 0
        count = 0
        for i in range(N):
            t = self.runEpisode()
            if t < 500:
                T += t
                count += 1
            # G += reward
        return T/count if count > 0 else 0.0
    
    def runEpisode(self, ):
        reward = 0
        state = self.d0()
        # t= 0.0
        for t in range(1000):
            action = self.pi_func(self.test_pol, state)
            # print(action)
            R = self.reward_function(state)
            next_state = self.trans_func(state, action)
            reward += (self.gamma**t) * R
            if next_state[0] == 0.5:
                break
            state = next_state
            t += 1.0
        # print("t = {}".format(t))
        return t


    def e_soft_policy_update(self, s, eps):
        row = np.digitize([s[0]], self.bins_x)[0]
        col = np.digitize([s[1]], self.bins_v)[0]

        best_a_list = []
        best_qsa = -float("inf")
        
        for i, expl_a in enumerate(self.actions):
            if best_qsa < self.q[row][col][i]:
                best_qsa = self.q[row][col][i]
                best_a_list = [i]
            elif best_qsa == self.q[row][col][i]:
                best_a_list.append(i)

        not_best_list = list(set(range(len(self.actions))) - set(best_a_list))
        new_prob = max(0, ((1- eps)/len(best_a_list)) + (eps/len(self.actions)))
        remaining_prob = (eps/len(self.actions))
        np.put(self.test_pol[row][col], best_a_list, [new_prob]*len(best_a_list))
        np.put(self.test_pol[row][col], not_best_list, [remaining_prob]*len(not_best_list))

    def softmax_policy_update(self, s, sigma):
        row = np.digitize([s[0]], self.bins_x)[0]
        col = np.digitize([s[1]], self.bins_v)[0]
        p = sigma*self.q[row][col]
        self.test_pol[row][col] = np.exp(p - max(p))/sum(np.exp(p - max(p)))


    def update_q(self, s, s_prime, a, r, alpha):
        # x_t_prime = np.digitize([x_t_prime], self.bins_x)[0]
        # v_t_prime =

        row = np.digitize([s[0]], self.bins_x)[0]
        col = np.digitize([s[1]], self.bins_v)[0]
        index_a = self.actions.index(a)
        # next_a = self.pi_func(pi=self.test_pol, s=s_prime)
        next_row = np.digitize([s_prime[0]], self.bins_x)[0]  
        next_col = np.digitize([s_prime[1]], self.bins_v)[0]
        # index_next_a = self.actions.index(next_a)
        self.q[row][col][index_a] = self.q[row][col][index_a] \
            + alpha * (r + (self.gamma * np.amax(self.q[next_row][next_col])) - self.q[row][col][index_a]) 


    def qlearning(self, eps):
        count = 0
        num_actions = 0
        num_episodes_list = []
        num_actions_list = []
        num_steps_list = []
        while True:
            count += 1
            s = self.d0()
            num_steps = 0
            while (s[0] != 0.5):
                num_steps += 1
                num_actions += 1
                a = self.pi_func(self.test_pol, s)
                s_prime = self.trans_func(s, a)
                r = self.reward_function(s)

                self.update_q(s, s_prime, a, r, self.alpha)

                # self.e_soft_policy_update(s=s, eps=min(eps, 1/count))
                self.softmax_policy_update(s=s, sigma=count)

                s = s_prime
            # print()
            # print("episode = {}, num actions till now = {}, num steps for this episode = {}".format(count, num_actions, num_steps))
            num_episodes_list.append(count)
            num_actions_list.append(num_actions)
            num_steps_list.append(num_steps)
            
            self.v = np.max(self.q, axis=2)

            if count > 150:
                break
                
        return count, num_episodes_list, num_actions_list, num_steps_list


    def model_update(self, s, a, s_prime, r):
        row = np.digitize([s[0]], self.bins_x)[0]
        col = np.digitize([s[1]], self.bins_v)[0]
        a_index = self.actions.index(a)
        self.model[row][col][a_index] = (r, s_prime)
        
    def get_rs_from_model(self, s, a):
        row = np.digitize([s[0]], self.bins_x)[0]
        col = np.digitize([s[1]], self.bins_v)[0]
        a_index = self.actions.index(a)
        return self.model[row][col][a_index]

    def leading_to_s(self, s):
        return self.predecessor[s] 

    def update_leading_to_s(self, s_, a_, s):
        if (s_, a_) not in self.predecessor[s]:
            self.predecessor[s].append((s_, a_))

    def get_priority(self, s, s_prime, r, a):
        row = np.digitize([s[0]], self.bins_x)[0]
        col = np.digitize([s[1]], self.bins_v)[0]
        index_a = self.actions.index(a)
        next_row = np.digitize([s_prime[0]], self.bins_x)[0]  
        next_col = np.digitize([s_prime[1]], self.bins_v)[0]

        priority = abs(r + (self.gamma * np.amax(self.q[next_row][next_col])) - self.q[row][col][index_a])
        return priority


    def prio_sweep(self, eps, threshold, n_iters, break_iters, running_average_length):
        count = 0
        num_actions = 0
        num_episodes_list = []
        num_actions_list = []
        alpha = self.alpha
        prev_v_list = []
        num_steps_list = []
        while True:
            # print(count)
            count += 1
            # alpha = max(alpha*0.999, 0.0001)
            num_steps = 0
            if len(prev_v_list) > running_average_length - 1:
                prev_v_list.pop(0)
            prev_v_list.append(copy.deepcopy(self.v))

            s = self.d0()
            while (s[0] != 0.5):
                num_steps += 1
                num_actions += 1
                a = self.pi_func(self.test_pol, s)
                s_prime = self.trans_func(s, a)
                self.update_leading_to_s(s, a, s_prime)

                r = self.reward_function(s)

                self.model_update(s, a, s_prime, r)

                priority = self.get_priority(s, s_prime, r, a)

                if priority > threshold:
                    self.pq.put((-priority, s, a))

                s = s_prime

                for itr_count in range(n_iters):
                    if self.pq.empty():
                        break
                    _, s1, a1 = self.pq.get()

                    r1, s_prime1 = self.get_rs_from_model(s1, a1)
                    self.update_q(s1, s_prime1, a1, r1, alpha)

                    # eps=max(eps*0.999, 0.0001)

                    # self.e_soft_policy_update(s=s1, eps=eps)
                    self.softmax_policy_update(s=s1, sigma=count)

                    for s_, a_ in self.leading_to_s(s1):
                        r_, _ = self.get_rs_from_model(s_, a_)
                        
                        priority_ = self.get_priority(s_, s1, r_, a_)
                        
                        if priority_ > threshold:
                            self.pq.put((-priority_, s_, a_))
            
            num_episodes_list.append(count)
            num_actions_list.append(num_actions)
            num_steps_list.append(num_steps)

            self.v = np.max(self.q, axis=2)

            if count > break_iters:
                break
    
        return count, num_episodes_list, num_actions_list, num_steps_list
    


def main():
    gamma = 0.9
    alpha = 0.05
    delta = 0.1
    sigma = 1
    num_bins = 4
    eps = 0.05

    # gamma = 0.9
    # alpha = 0.1
    # delta = 0.1
    # sigma = 1
    threshold = 1e-5
    # eps = 0.1
    n_iters = 10
    break_iters = 60
    running_average_length = 200

    for num_bins in range(4, 5):
        num_acts_mean_list = []
        num_steps_mean = []
        J = []
        for _ in tqdm(range(2)):
            ps = PrioritizedSweeping(alpha=alpha, gamma=gamma, delta=delta, num_bins=num_bins)
            count, num_episodes_list, num_actions_list, num_steps_list = ps.prio_sweep(eps=eps, threshold=threshold, n_iters=n_iters, break_iters=break_iters, running_average_length=running_average_length)
            time = ps.estimate_J(20)
            if time > 0.0:
                J.append(time)
            # print()
            # plt.figure(0)
            # plt.plot(num_actions_list, num_episodes_list, 'c')
            # plt.title("Learning curve")
            # plt.xlabel("Number of Steps till now")
            # plt.ylabel("Episodes")
            num_acts_mean_list.append(num_actions_list)

            # plt.figure(1)
            # plt.plot(num_episodes_list, num_steps_list, 'c')
            # plt.title("Learning curve")
            # plt.xlabel("Time Steps")
            # plt.ylabel("Episodes")
            num_steps_mean.append(num_steps_list)

        column_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*num_acts_mean_list)]

        plt.figure(0)
        plt.plot(column_average, num_episodes_list, label=str(int(num_bins)))
        # plt.plot(column_average, num_episodes_list, 'k')
        plt.title("Learning curve")
        plt.xlabel("Number of Steps till now")
        plt.ylabel("Episodes")
        # eight = mlines.Line2D([], [], color='c', marker='s', ls='', label='')
        # nine = mlines.Line2D([], [], color='k', marker='s', ls='', label='mean')
        # plt.legend(handles=[nine])
        plt.legend()

        column_average_steps = [sum(sub_list) / len(sub_list) for sub_list in zip(*num_steps_mean)]
        # column_variance_steps = [statistics.stdev(i) for i in zip(*num_steps_mean)]

        plt.figure(1)
        # plt.plot(num_episodes_list, column_average_steps, 'k')
        plt.plot(num_episodes_list, column_average_steps, label=str(int(num_bins)))
        # plt.fill_between(column_average_steps, list(map(float.__add__, column_average_steps, column_variance_steps)), list(map(float.__sub__, column_average_steps, column_variance_steps)), facecolor='blue', alpha=0.5)
        # plt.errorbar(num_episodes_list, column_average_steps, yerr=column_variance_steps, elinewidth=0.75, linestyle='None')
        plt.title("Learning curve")
        plt.ylabel("Number of Steps")
        plt.xlabel("Episodes")
        # eight = mlines.Line2D([], [], color='c', marker='s', ls='', label='')
        # nine = mlines.Line2D([], [], color='k', marker='s', ls='', label='mean')
        # plt.legend(handles=[nine])
        plt.legend()
        # print("average time to reach target with learned policy = {}".format(mean(J)))
    plt.show()



    # best_threshold = None
    # best_alpha = None
    # best_eps = None
    # best_sigma = None
    # best_v = None
    # best_n_iters = None
    # best_max_norm = float("inf")
    # for best_sigma in np.arange(1, 2, 1):
    #     ps = PrioritizedSweeping(alpha=alpha, gamma=gamma, delta=delta, num_bins=num_bins)
    #     count, num_episodes_list, num_actions_list, num_steps_list = ps.prio_sweep(eps=eps, threshold=threshold, n_iters=n_iters, break_iters=break_iters, running_average_length=running_average_length)
    # #     if best_max_norm > max_norm:
    # #         best_max_norm = max_norm
    # #         best_threshold = threshold
    # #         best_sigma = sigma
    # #         best_alpha = alpha
    # #         best_eps = eps
    # #         best_v = ps.v
    # #         best_n_iters = n_iters
    # #         best_break_iters = break_iters
    # #         ps.print_policy()
    # #     # print(ps.v)
    # # print("alpha = {}, eps = {}, threshold = {}, sigma= {}, n_iters = {}, break_iters = {} max_norm = {}".format(best_alpha, best_eps, best_threshold, best_sigma, best_n_iters, break_iters, best_max_norm))
    # # print(best_v)


if __name__ == '__main__':
    main()