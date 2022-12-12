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
from queue import PriorityQueue
import gym

class PrioritizedSweeping():
    def __init__(self, alpha, gamma, delta):
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta



        self.model = defaultdict(tuple)
        self.pq = PriorityQueue()
        self.predecessor = defaultdict(list)
        
        self.env = gym.make("Taxi-v3")
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        self.q = np.array([[0.0 for a in range(self.action_size)] for j in range(self.state_size)])
        self.test_pol = np.array([[1/self.action_size for a in range(self.action_size)] for j in range(self.state_size)])
        pass
    

    def pi_func(self, pi, s):
        """
        Args: 
            pi: dictionary key = states, value = list of actions
            s = tuple(row, col)
            eps = float
        Returns:
            action in state s (tuple)
        """
        chosen_action = np.random.choice(self.action_size, 1, p=pi[s])
        return chosen_action[0]


    def e_soft_policy_update(self, s, eps):
        best_a_list = []
        best_qsa = -float("inf")
        for i in range(self.action_size):
            if best_qsa < self.q[s][i]:
                best_qsa = self.q[s][i]
                best_a_list = [i]
            elif best_qsa == self.q[s][i]:
                best_a_list.append(i)
        not_best_list = list(set(range(self.action_size)) - set(best_a_list))
        new_prob = max(0, ((1- eps)/len(best_a_list)) + (eps/self.action_size))
        remaining_prob = (eps/self.action_size)
        np.put(self.test_pol[s], best_a_list, [new_prob]*len(best_a_list))
        np.put(self.test_pol[s], not_best_list, [remaining_prob]*len(not_best_list))


    def model_update(self, s, a, s_prime, r):
        self.model[(s, a)] = (r, s_prime)
        
    def get_rs_from_model(self, s, a):
        return self.model[(s, a)]

    def leading_to_s(self, s):
        return self.predecessor[s] 

    def update_leading_to_s(self, s_, a_, s):
        if (s_, a_) not in self.predecessor[s]:
            self.predecessor[s].append((s_, a_))

    def prio_sweep(self, eps, threshold, n_iters, break_iters, running_average_length):
        max_norm = []
        mse = []
        itr_number = []
        count = 0
        
        num_episodes_list = []
        num_actions_list = []
        alpha = self.alpha
        prev_v_list = []
        reward_acc = []
        cumm_action_list = []
        cumm_action = 0
        while True:
            self.predecessor = defaultdict(list)
            count += 1
            # alpha = max(alpha*0.999, 0.0001)
            reward = 0
            s = self.env.reset()
            isterminal = False
            # print(s)
            num_actions = 0
            while not isterminal:
                cumm_action += 1
                num_actions += 1

                a = self.pi_func(self.test_pol, s)
                s_prime, r, isterminal, _= self.env.step(a)
                self.update_leading_to_s(s, a, s_prime)
                self.model_update(s, a, s_prime, r)
                reward += r

                priority = abs(r + (self.gamma * np.amax(self.q[s_prime])) - self.q[s][a])
                if priority > threshold:
                    self.pq.put((-priority, s, a))

                s = s_prime

                for itr_count in range(n_iters):

                    if self.pq.empty():
                        break

                    _, s1, a1 = self.pq.get()
                    r1, s_prime1 = self.get_rs_from_model(s1, a1)

                    self.q[s1][a1] = self.q[s1][a1] \
                        + alpha * (r1 + (self.gamma * np.amax(self.q[s_prime1])) - self.q[s1][a1]) 

                    eps=max(eps*0.999, 0.0001)
                    self.e_soft_policy_update(s=s1, eps=eps)

                    for s_, a_ in self.leading_to_s(s1):
                        r_, _ = self.get_rs_from_model(s_, a_)
                        priority_ = abs(r_ + (self.gamma * np.amax(self.q[s1])) - self.q[s_][a_])

                        if priority_ > threshold:
                            self.pq.put((-priority_, s_, a_))

                

            num_episodes_list.append(count)
            num_actions_list.append(num_actions)
            reward_acc.append(reward)
            cumm_action_list.append(cumm_action)
            if count > break_iters:
                break
        return count, num_episodes_list, num_actions_list, reward_acc, cumm_action_list

    def test_policy(self, N):
        reward_mean = []
        num_actions_mean = []
        for i in range(N):
            s = self.env.reset()
            isterminal = False
            num_actions = 0
            reward = 0
            while not isterminal:
                num_actions += 1
                a = self.pi_func(self.test_pol, s)
                s_prime, r, isterminal, _= self.env.step(a)
                s = s_prime
                reward += r
            reward_mean.append(reward)
            num_actions_mean.append(num_actions)
        return mean(reward_mean), mean(num_actions_mean)

    def mse(self, m1, m2):
        return np.square(np.subtract(m1, m2)).mean() 

    def print_policy(self):
        print("Policy:") # Printing policy
        k = 0
        for i in range(5):
            for j in range(5):
                if i == 4 and j == 4:
                    print("G", end = " ")
                elif (i == 2 or i == 3) and j == 2:
                    print(" ", end = " ")
                else:
                    ind = np.argmax(self.test_pol[i][j])
                    print(self.arrows[ind], end=" ")
                    k += 1
            print()

    def plot_figures_from_returns_array(self, all_returns, x_label, y_label, title):
        returns_mean= np.average(all_returns,axis=0)
        std_err = np.std(all_returns,axis=0)
        x_arr = np.arange(len(returns_mean)) + 1

        plt.figure(figsize=(9,5))
        plt.plot(x_arr, returns_mean, '-', color='gray')
        # plt.plot(x_arr, returns_mean)
        plt.errorbar(x_arr, returns_mean, yerr=std_err, fmt='o', color='black',
                    ecolor='lightgray', elinewidth=3, capsize=0);
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()

def main():
    gamma = 0.9
    alpha = 0.2
    delta = 0.1
    sigma = 1
    threshold = 0.5 #1e-9
    eps = 0.01
    n_iters = 20
    break_iters = 500
    running_average_length = 300
    best_reward = -float("inf")
    all_returns = []
    all_actions = []
    all_cumm_actions = []
    for _ in range(2):
        ps = PrioritizedSweeping(alpha=alpha, gamma=gamma, delta=delta)
        count, num_episodes_list, num_actions_list, reward_acc, cumm_action_list = ps.prio_sweep(eps=eps, threshold=threshold, n_iters=n_iters, break_iters=break_iters, running_average_length=running_average_length)
        all_returns.append(reward_acc)
        all_actions.append(num_actions_list)
        all_cumm_actions.append(cumm_action_list)
        mean_r, mean_actions = ps.test_policy(10)
        print("Test time evaluation of policy: mean_r = {}, mean_actions = {}".format(mean_r, mean_actions))

        if best_reward < mean_r:
            best_reward = mean_r


    ps.plot_figures_from_returns_array(all_returns, "Number of Episodes","Average Rewards", "Learning curve for Taxi MDP for Prioritized sweeping")
    ps.plot_figures_from_returns_array(all_actions, "Number of Episodes","Average steps to reach goal", "Learning curve for Taxi MDP for Prioritized sweeping")


    cumm_actions_mean= np.average(all_cumm_actions,axis=0)
    std_err = np.std(all_cumm_actions,axis=0)
    x_arr = np.arange(len(cumm_actions_mean)) + 1
    plt.figure(figsize=(9,5))
    plt.plot(cumm_actions_mean, x_arr)
    plt.xlabel("Total steps to reach goal")
    plt.ylabel("Number of Episodes")
    plt.title("Learning curve for Taxi MDP for Prioritized sweeping")
    plt.show()

    
if __name__ == '__main__':
    main()
