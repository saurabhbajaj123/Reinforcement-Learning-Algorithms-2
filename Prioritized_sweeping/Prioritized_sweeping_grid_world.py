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

class PrioritizedSweeping():
    def __init__(self, num_rows, num_cols, alpha, gamma, delta):
        self.actions = [(-1,0), (0,1), (1,0), (0,-1)] # up, right, down, left 
        self.arrows = ["↑", "→","↓", "←"]
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta


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

        # self.pi_esoft = collections.defaultdict(list) # key: state, value: list(best action)
        # for s in self.states:
        #     self.pi_esoft[s] = self.actions


        self.states = [(i, j) for j in range(self.num_cols) for i in range(self.num_rows)]
        self.v = np.array([[0.0 for j in range(self.num_cols)] for i in range(self.num_rows)])
        # self.policy = np.array([[(0, 1) for j in range(self.num_cols)] for i in range(self.num_rows)])
        self.test_pol = np.array([[[1/len(self.actions) for k in range(len(self.actions))] for j in range(self.num_cols)] for i in range(self.num_rows)])
        self.q = np.array([[[0.0 for a in range(len(self.actions))] for j in range(self.num_cols)] for i in range(self.num_rows)])
        
        self.model = np.array([[[(0.0, (0, 0)) for a in range(len(self.actions))] for j in range(self.num_cols)] for i in range(self.num_rows)]) # this is a 3D matrix, with each value being a pair (reward, state)
        self.pq = PriorityQueue()
        self.predecessor = defaultdict(list, {k : [] for k in self.states})
        pass
    
    def transition_function(self):
        counter = 0
        p = defaultdict(list)
        for state in self.states: 
            for next_direction in [(-1,0), (0,1), (1,0), (0,-1), (0,0)]:
                # print("next_direction = {}".format(next_direction[0]))
                # print("state = {}".format(state))
                next_state = (state[0] + next_direction[0], state[1] + next_direction[1]) 
                # print(next_state)
                # print()
                for action in self.actions:
                    
                    prob = 0
                    if ((next_state[0] < 0) or (next_state[1] < 0) or (next_state[0] > 4) or (next_state[1] > 4)):
                        continue
                    if ((state[0] == 2) and (state[1] == 2)) or ((state[0] == 3) and (state[1] == 2)):
                        prob = 0
                        p[state, next_state].append(prob)
                        continue
                    if ((next_state[0] == 2) and (next_state[1] == 2)) or ((next_state[0] == 3) and (next_state[1] == 2)):
                        prob = 0
                        p[state, next_state].append(prob)
                        continue
                    if ((state[0] == 4) and (state[1] == 4)):
                        if ((next_state[0] == 4) and (next_state[1] == 4)):
                            prob = 1
                        else:
                            prob = 0
                        p[state, next_state].append(prob)
                        continue
                    # if ((state[0] == 0) and (state[1] == 2)):
                    #     if ((next_state[0] == 0) and (next_state[1] == 2)):
                    #         prob = 1
                    #     else:
                    #         prob = 0
                    #     p[state, next_state].append(prob)
                    #     continue
                    if action == next_direction:
                        prob = 0.8
                    elif (next_direction == self.actions[(self.actions.index(action) + 1) % 4]) or (next_direction == self.actions[(self.actions.index(action) - 1) % 4]):
                        prob = 0.05
                    elif (next_direction == (0,0)):
                        prob = 0.1
                        if ((state[0] + action[0]) == 2 and (state[1] + action[1]) == 2): # going into the obstacle directly when moving towards the obstacle
                            prob += 0.8
                        if ((state[0] + action[0]) == 3 and (state[1] + action[1]) == 2):# going into the obstacle directly when moving towards the obstacle
                            prob += 0.8
                        if (((state[0] + action[0]) < 0)  or ((state[1] + action[1]) < 0) or ((state[0] + action[0]) > 4)  or ((state[1] + action[1]) > 4)): # going directly into the wall
                            prob += 0.8
                        if (((state[0] + self.actions[(self.actions.index(action) + 1) % 4][0]) == 2) and ((state[1] + self.actions[(self.actions.index(action) + 1) % 4][1]) == 2)): # going into the obstacle mistakenly towrads right
                            prob += 0.05
                        if (((state[0] + self.actions[(self.actions.index(action) + 1) % 4][0]) == 3) and ((state[1] + self.actions[(self.actions.index(action) + 1) % 4][1]) == 2)):# going into the obstacle 
                            prob += 0.05
                        if (((state[0] + self.actions[(self.actions.index(action) - 1) % 4][0]) == 2) and ((state[1] + self.actions[(self.actions.index(action) - 1) % 4][1]) == 2)): # going into the obstacle
                            prob += 0.05
                        if (((state[0] + self.actions[(self.actions.index(action) - 1) % 4][0]) == 3) and ((state[1] + self.actions[(self.actions.index(action) - 1) % 4][1]) == 2)):# going into the obstacle
                            prob += 0.05
                        if ((state[0] + self.actions[(self.actions.index(action) + 1) % 4][0]) < 0): # going into the wall
                            prob += 0.05
                        if ((state[1] + self.actions[(self.actions.index(action) + 1) % 4][1]) < 0): # going into the wall
                            prob += 0.05
                        if ((state[0] + self.actions[(self.actions.index(action) + 1) % 4][0]) > 4): # going into the wall
                            prob += 0.05
                        if ((state[1] + self.actions[(self.actions.index(action) + 1) % 4][1]) > 4): # going into the wall
                            prob += 0.05
                        if ((state[0] + self.actions[(self.actions.index(action) - 1) % 4][0]) < 0): # going into the wall
                            prob += 0.05
                        if ((state[1] + self.actions[(self.actions.index(action) - 1) % 4][1]) < 0): # going into the wall
                            prob += 0.05
                        if ((state[0] + self.actions[(self.actions.index(action) - 1) % 4][0]) > 4): # going into the wall
                            prob += 0.05
                        if ((state[1] + self.actions[(self.actions.index(action) - 1) % 4][1]) > 4): # going into the wall
                            prob += 0.05
                    
                    # print("state = {}, action = {}, next_state = {}, prob = {}".format(state, action, next_state, round(prob, 3)))
                    p[state, next_state].append(round(prob, 3))
        # print(len(p))
        return p
    
    
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
        s_prime = (s[0] + a[0], s[1] + a[1])

        # if rand < 0.8:
        #     s_prime = (s[0] + a[0], s[1] + a[1])
        # elif 0.8 < rand <  0.85:
        #     a = self.actions[(self.actions.index(a) + 1) % 4]
        #     s_prime = (s[0] + a[0], s[1] + a[1])
        # elif  0.85 < rand <  0.9:
        #     a = self.actions[(self.actions.index(a) - 1) % 4]
        #     s_prime = (s[0] + a[0], s[1] + a[1])
        # else:
        #     s_prime = s
        if (s_prime == (2,2)) or (s_prime == (3,2)) or (s_prime[0] < 0) or (s_prime[0] > 4) or (s_prime[1] < 0) or (s_prime[1] > 4):
            s_prime = s
        return s_prime





    def reward(self, s, a, s_prime):
        if (s == (4, 4)):
            return 0
        elif s_prime == (4, 4):
            return 10
        elif s_prime == (4, 2):
            return -10
        # elif s_prime == (0, 2):
        #     # return 5
        #     return 4.4844 # found using binary search
        else:
            return 0

    def d0(self):
        states = self.states.copy()
        states.remove((2,2))
        states.remove((3,2))
        # states.remove((4,4))
        random_index = random.randint(0,len(states)-1)
        return states[random_index]




    def run(self, threshold):
        # self.v = np.array([[0 for j in range(5)] for i in range(5)])
        p = self.transition_function()
        count = 0
        while True:
            count += 1
            delta = 0
            v_old = np.copy(self.v)
            # print(np.amax(v_old))
            for s in self.states:
                max_val = -float("inf")
                max_a = None
                for i, a in enumerate(self.actions):
                    val = 0
                    for s_prime in self.states:
                        
                        # print(s, s_prime)
                        try:
                            # print(v_old[s_prime])
                            val += p[s, s_prime][i]*(self.reward(s, a, s_prime) + (self.gamma*v_old[s_prime]))
                        except:
                            continue
                    # print("val = {}".format(val))
                    if max_val < val:
                        max_val = val
                        max_a = i
                # if (s == (1, 1)): print("val = {}".format(val))
                self.v[s] = round(max_val, 4)
                self.policy[s] = self.actions[max_a]
            delta = max(delta, np.amax(abs(self.v - v_old)))
            if delta < threshold:
                break
        return self.v, self.policy, count

    def pi_func(self, pi, s):
        """
        Returns probability
        """
        # self.gamma = 0.9
        # v_star, pi_star, iterations = self.run(0.0001)

        return pi[s[0]][s[1]]


    def v_star_func(self, s):
        # self.gamma = 0.9
        # v_star, pi_star, iterations = self.run(0.0001)
        return self.v_star[s[0]][s[1]]

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

    def pi_esoft_func(self, pi, s, eps):
        """
        Args: 
            pi: dictionary key = states, value = list of actions
            s = tuple(row, col)
            eps = float
        Returns:
            action in state s (tuple)
        """
        # rand = random.uniform(0, 1)
        # A_star = pi[s]
        # A = self.actions
        # A_ = list(set(A) - set(A_star))
        # # print(A_star, A, A_)
        # prob = ((1- eps)/len(A_star)) + (eps/len(A))
        # for i in range(len(A_star)):
        #     if prob*(i) < rand < prob*(i+1):
        #         return A_star[i]
        # for i in range(len(A_)):
        #     if (prob*len(A_star) + (eps/len(A))*(i)) < rand < (prob*len(A_star) + (eps/len(A))*(i+1)):
        #         return A_[i]
        # print(s)
        chosen_action = np.random.choice(4, 1, p=pi[s[0]][s[1]])
        return self.actions[chosen_action[0]]



    def policy_prob(self, s, a, pi, eps):
        A_star = pi[s]
        A = self.actions
        # print(a, A_star)
        if a in A_star:
            # print(((1- eps)/len(A_star)) + (eps/len(A)))
            return ((1- eps)/len(A_star)) + (eps/len(A))
        else:
            # print("else condition")
            # print((eps/len(A)))
            return (eps/len(A))

    def generateEpisode_esoft(self, pi, eps):
        """
        Args:
            pi = dictionary key = states, value = list of actions
            eps
        """
        trajectory = []
        s = self.d0()
        while(s != (4, 4)):
            a = self.pi_esoft_func(pi, s, eps)
            s_prime = self.trans_func(s, a)
            r = self.reward(s, a, s_prime)
            trajectory.append(((s, a), r))
            s = s_prime
        trajectory.append((((4,4), (0, 1)), 0))
        return trajectory
    
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


    def model_update(self, s, a, s_prime, r):
        a_index = self.actions.index(a)
        self.model[s[0]][s[1]][a_index] = (r, s_prime)
        
    def get_rs_from_model(self, s, a):
        a_index = self.actions.index(a)
        return self.model[s[0]][s[1]][a_index]

    def leading_to_s(self, s):
        return self.predecessor[s] 

    def update_leading_to_s(self, s_, a_, s):
        if (s_, a_) not in self.predecessor[s]:
            self.predecessor[s].append((s_, a_))
        # print(self.predecessor[s])

    def prio_sweep(self, eps, threshold, n_iters, break_iters, running_average_length):
        max_norm = []
        mse = []
        itr_number = []
        count = 0
        num_actions = 0
        num_episodes_list = []
        num_actions_list = []
        alpha = self.alpha
        prev_v_list = []
        while True:
            # print("itr number = {}".format(count))
            count += 1
            alpha = max(alpha*0.999, 0.0001)
            prev_q = copy.deepcopy(self.q)
            prev_v = copy.deepcopy(self.v)

            if len(prev_v_list) > running_average_length - 1:
                prev_v_list.pop(0)
            prev_v_list.append(copy.deepcopy(self.v))
            # if count % 5 == 0:
            #     self.predecessor = defaultdict(list, {k : [] for k in self.states})

            s = self.d0()
            # print(s)
            while s != (4, 4):
                # print("state = {}".format(s))
                a = self.pi_esoft_func(self.test_pol, s, eps)
                s_prime = self.trans_func(s, a)
                self.update_leading_to_s(s, a, s_prime)
                r = self.reward(s, a, s_prime)

                self.model_update(s, a, s_prime, r)
                

                row = s[0]
                col = s[1]
                index_a = self.actions.index(a)
                next_row = s_prime[0] 
                next_col = s_prime[1]

                priority = abs(r + (self.gamma * np.amax(self.q[next_row][next_col])) - self.q[row][col][index_a])
                # print("priority = {}".format(priority))
                if priority > threshold:
                    # print(-priority, s, a)
                    self.pq.put((-priority, s, a))

                s = s_prime

                # while not self.pq.empty():
                for itr_count in range(n_iters):

                    if self.pq.empty():
                        break
                    # print("itr_count = {}".format(itr_count))
                    # print("self.pq = {}".format(self.pq))

                    _, s1, a1 = self.pq.get()

                    # print("s1 = {}, a1 = {}".format(s1, a1))
                    r1, s_prime1 = self.get_rs_from_model(s1, a1)
                    # print("entering for loop")

                    row = s1[0]
                    col = s1[1]
                    index_a = self.actions.index(a1)
                    next_row = s_prime1[0] 
                    next_col = s_prime1[1]
                    # if row == 2 and col == 0: 
                    #     print(self.q[row][col])
                    #     print(self.test_pol[row][col])
                    # if row == 4 and col == 1: print(self.q[row][col][index_a])
                    # if row == 4 and col == 0: print(self.q[row][col][index_a])

                    self.q[row][col][index_a] = self.q[row][col][index_a] \
                        + alpha * (r1 + (self.gamma * np.amax(self.q[next_row][next_col])) - self.q[row][col][index_a]) 

                    eps=max(eps*0.999, 0.0001)
                    self.e_soft_policy_update(s=s1, eps=eps)
                    # self.softmax_policy_update(s=s1, sigma=count)

                    for s_, a_ in self.leading_to_s(s1):
                        # print("s_ = {}, a_ = {}".format(s_, a_))
                        # r_ = self.reward(s_, a_, s1)
                        r_, _ = self.get_rs_from_model(s_, a_)

                        row_ = s_[0]
                        col_ = s_[1]
                        index_a_ = self.actions.index(a_)
                        next_row_ = s1[0]
                        next_col_ = s1[1]

                        priority_ = abs(r_ + (self.gamma * np.amax(self.q[next_row_][next_col_])) - self.q[row_][col_][index_a_])
                        # print(priority_, s_, a_, s1)
                        if priority_ > threshold:
                            self.pq.put((-priority_, s_, a_))

                

            num_episodes_list.append(count)
            num_actions_list.append(num_actions)
            if count % 250 == 0:
                mse.append(self.mse(self.v, self.v_star))
                itr_number.append(count)
            if count > break_iters:
                break
            # for s in self.states:
            #     self.v[s[0]][s[1]] = sum([self.test_pol[s[0]][s[1]][a_index]*self.q[s[0]][s[1]][a_index] for a_index, a in enumerate(self.actions)])
            self.v = np.max(self.q, axis=2)

            # print("shape = {}".format((self.v).shape))
            # if np.amax(abs(self.v - prev_v)) < self.delta:
            #     break            
            max_norm.append(np.amax(abs(self.v - self.v_star)))
            if np.amax(abs(self.v - self.v_star)) < self.delta:
                break
            if np.amax(abs(self.v - np.mean(prev_v_list, axis=0))) < self.delta:
                break
            plt.plot(max_norm)
            plt.title("Max norm")
            plt.xlabel("Iterations")
            plt.ylabel("Max norm")
            plt.pause(0.0001)


            # num_acts_mean_list.append(num_actions_list)
            # plt.figure(0)
            # plt.plot(num_actions_list, num_episodes_list, 'c')
            # plt.title("Learning curve")
            # plt.xlabel("Time Steps")
            # plt.ylabel("Episodes")

            # mse_mean_list.append(mse)
            # plt.figure(1)
            # plt.plot(itr_number, mse, 'r')
            # plt.title("Mean squared Error")
            # plt.xlabel("Iterations")
            # plt.ylabel("MSE")

        # column_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*num_acts_mean_list)]
        # plt.figure(0)
        # plt.plot(column_a
        # verage, num_episodes_list, 'k')
        # plt.title("Learning curve")
        # plt.xlabel("Time Steps")
        # plt.ylabel("Episodes")
        # # eight = mlines.Line2D([], [], color='c', marker='s', ls='', label='')
        # nine = mlines.Line2D([], [], color='k', marker='s', ls='', label='mean')
        # plt.legend(handles=[nine]) 



        # print(column_average_mse)
        # column_average_mse = [sum(sub_list) / len(sub_list) for sub_list in zip(*mse_mean_list)]
        # plt.figure(1)
        # plt.plot(itr_number, column_average_mse, 'k')
        # plt.title("Mean squared Error")
        # plt.xlabel("Iterations")
        # plt.ylabel("MSE")
        # nine = mlines.Line2D([], [], color='k', marker='s', ls='', label='mean')
        # plt.legend(handles=[nine])



        # # plt.plot(itr_number, mse)
        # # # plt.title("Mean squared Error for eps = {}".format(eps))
        # # plt.title("Mean squared Error for alpha = {}".format(self.alpha))
        # # plt.xlabel("Iterations")
        # # plt.ylabel("MSE")
        # plt.pause(0.0001)
        # plt.show()
        return count, num_episodes_list, num_actions_list, mse, itr_number, max_norm[-1]
    


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


def main():
    def replace(inp, positions, char):
        for pos in positions:
            inp[pos] = char
    obstacles = [(2,2), (3,2)]
    goal = [(0,2), (4,4)]

    # gamma = 0.9
    # alpha = 0.1
    # delta = 0.1
    # sigma = 1
    # threshold = 0.0
    # eps = 0.1
    # n_iters = 10
    # print("running Q Learning")
    num_acts_mean_list = []
    mse_mean_list = []
    # for _ in tqdm(range(20)):
    #     ps = PrioritizedSweeping(alpha=alpha, gamma=gamma, delta=delta)
    #     count, num_episodes_list, num_actions_list, mse, itr_number = ps.prio_sweep(eps=sigma, threshold=threshold)

    #     plt.figure(0)
    #     plt.plot(num_actions_list, num_episodes_list, 'c')
    #     plt.title("Learning curve")
    #     plt.xlabel("Number of Steps")
    #     plt.ylabel("Episodes")

    #     plt.figure(1)
    #     plt.plot(itr_number, mse, 'r')
    #     plt.title("Mean squared Error")
    #     plt.xlabel("Iterations")
    #     plt.ylabel("MSE")

    #     num_acts_mean_list.append(num_actions_list)
    #     mse_mean_list.append(mse)

    #     # sarsa.print_policy()


    # column_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*num_acts_mean_list)]
    # plt.figure(0)
    # plt.plot(column_average, num_episodes_list, 'k')
    # plt.title("Learning curve")
    # plt.xlabel("Time Steps")
    # plt.ylabel("Episodes")
    # # eight = mlines.Line2D([], [], color='c', marker='s', ls='', label='')
    # nine = mlines.Line2D([], [], color='k', marker='s', ls='', label='mean')
    # plt.legend(handles=[nine])



    # column_average_mse = [sum(sub_list) / len(sub_list) for sub_list in zip(*mse_mean_list)]
    # plt.figure(1)
    # plt.plot(itr_number, column_average_mse, 'k')
    # plt.title("Mean squared Error")
    # plt.xlabel("Iterations")
    # plt.ylabel("MSE")
    # nine = mlines.Line2D([], [], color='k', marker='s', ls='', label='mean')
    # plt.legend(handles=[nine])

    # # print(sarsa.v)
    # plt.show()

    gamma = 0.9
    alpha = 0.1
    delta = 0.1
    sigma = 1
    threshold = 0.0 #1e-9
    eps = 0.1
    n_iters = 25
    break_iters = 500
    running_average_length = 200

    best_threshold = None
    best_alpha = None
    best_eps = None
    best_sigma = None
    best_v = None
    best_n_iters = None
    best_max_norm = float("inf")
    for best_sigma in np.arange(1, 5, 1):
        ps = PrioritizedSweeping(num_rows=5, num_cols=5, alpha=alpha, gamma=gamma, delta=delta)
        count, num_episodes_list, num_actions_list, mse, itr_number, max_norm = ps.prio_sweep(eps=eps, threshold=threshold, n_iters=n_iters, break_iters=break_iters, running_average_length=running_average_length)
        if best_max_norm > max_norm:
            best_max_norm = max_norm
            best_threshold = threshold
            best_sigma = sigma
            best_alpha = alpha
            best_eps = eps
            best_v = ps.v
            best_n_iters = n_iters
            best_break_iters = break_iters
            ps.print_policy()
        # print(ps.v)
    print("alpha = {}, eps = {}, threshold = {}, sigma= {}, n_iters = {}, break_iters = {} max_norm = {}".format(best_alpha, best_eps, best_threshold, best_sigma, best_n_iters, break_iters, best_max_norm))
    print(best_v)
if __name__ == '__main__':
    main()
