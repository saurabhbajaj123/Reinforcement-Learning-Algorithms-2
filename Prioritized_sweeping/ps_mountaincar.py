import numpy as np
from collections import defaultdict
import random
from matplotlib import pyplot as plt
import math
import copy
from queue import PriorityQueue

actions = {0 : -1, 1 : 0, 2 : 1}
actions_index = {-1:0, 0:1, 1:2 }
def discretize(state, bins):
    x, v = state
    x_index = (x - (-1.2))/(0.5-(-1.2))*bins
    v_index = (v - (-0.07))/(0.07-(-0.07))*bins
    if x_index == bins:
        x_index = bins-1
    if v_index == bins:
        v_index = bins - 1
    return (math.floor(x_index), math.floor(v_index))

# actions -> [0, 1, 2] = [-1, 0, 1]
def choose_action(q, s, epsilon):
    x = np.array(q[s])
    a_star = list(np.argwhere(x == max(x)).flatten())
    prob = np.random.uniform()
    if prob < epsilon:
        return random.sample([0, 1, 2], 1)[0]
    else:
        return random.sample(a_star, 1)[0]

def observe(s, a, no_of_bins):
    x_t, v_t = s
    next_v = v_t + 0.001 * actions[a] - 0.0025 * math.cos(3 * x_t)
    next_x = x_t + next_v

    if next_v < -0.07:
        next_v = -0.07
    if next_v > 0.07:
        next_v = 0.07

    if next_x < -1.2:
        next_x = -1.2
        next_v = 0
    if next_x > 0.5:
        next_x = 0.5
        next_v = 0

    reward = -1
    if discretize((next_x, next_v), no_of_bins)[0] == no_of_bins-1:
        reward = 10
    
    return (next_x, next_v), reward

def get_x(state, action, bins):
    x = np.zeros((bins * bins * 3))
    index = state[0]*(bins * 3) + state[1]*3 + action
    if (discretize(state, bins)[0] != bins - 1):
        x[index] = 1
    return x

def true_online_sarsa(gamma, alpha, lam, epsilon, max_iters, bins):
    iters = 0
    q = np.zeros((bins, bins, 3)) # q -> [["-1", "0", "1"]], 0 - -1, 1 - 0, 2 - 1
    w = np.random.rand((bins * bins * 3))
    steps_arr = []
    while True:
        # print(iters)
        if iters == max_iters:
            break
        # start an episode
        steps = 0
        start_state = (random.uniform(-0.6, -0.4), 0)
        s = start_state
        dis_s = discretize(s, bins)
        a = choose_action(q, dis_s, epsilon)
        x = get_x(dis_s, a, bins)
        Q_old = 0
        z = np.zeros((bins * bins * 3))
        while (dis_s[0] != bins - 1) and steps < 1001:
            steps += 1
            next_state, reward = observe(s, a, bins)
            dis_next_state = discretize(next_state, bins)
            next_action = choose_action(q, dis_next_state, epsilon)
            next_x = get_x(dis_next_state, next_action, bins)
            # print("x_shape: ")
            # print(x.shape)
            # print("w shape: ")
            # print(w.shape)
            Q = np.dot(w, x)
            next_Q = np.dot(w, next_x)
            # print("Q': " + str(next_Q))
            q[dis_next_state][next_action] = next_Q 
            delta = reward + gamma * next_Q - Q
            z = gamma * lam * z + (1 - alpha * gamma * lam * (z.T) * x) * x
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = next_Q
            x = next_x
            s = next_state
            dis_s = dis_next_state
            a = next_action
            # print(q)
            # q[s[0]][s[1]][a] += alpha * (reward + gamma * q[next_state[0]][next_state[1]][next_action] - q[s[0]][s[1]][a])

        steps_arr.append(steps)
        iters += 1
        

    return q, iters, steps_arr

def update_leading_to_s(predecessor, s_, a_, s):
    if (s_, a_) not in predecessor[s]:
        predecessor[s].append((s_, a_))
    return predecessor


def model_update(model, s, a, s_prime, r):
    # row = s[0]
    # col = s[1]
    # # print(a)
    # a_index = a
    # model[row][col][a_index] = (r, s_prime)
    model[(s, a)] = (r, s_prime)


def get_priority(q, s, s_prime, r, a):
    row = s[0]
    col = s[1]
    index_a = a
    next_row = s_prime[0]  
    next_col = s_prime[1]

    priority = abs(r + (gamma * np.amax(q[next_row][next_col])) - q[row][col][index_a])
    return priority

def get_rs_from_model(model, s, a):
    # row = s[0]
    # col = s[1]
    # a_index = a
    # return model[row][col][a_index]
    return model[(s, a)]


def update_q(q, s, s_prime, a, r, alpha, gamma):
    row = s[0]
    col = s[1]
    index_a = a
    next_row = s_prime[0]  
    next_col = s_prime[1]
    q[row][col][index_a] = q[row][col][index_a] \
        + alpha * (r + (gamma * np.amax(q[next_row][next_col])) - q[row][col][index_a]) 

def leading_to_s(predecessor, s):
    return predecessor[s] 

def prio_sweep(threshold, n_iters, gamma, alpha, epsilon, max_iters, bins):
    iters = 0
    q = np.zeros((bins, bins, 3))
    pq = PriorityQueue()
    states = [(i, j) for j in range(bins) for i in range(bins)]
    # predecessor = defaultdict(list, {k : [] for k in states})
    # model = np.array([[[(0.0, (0, 0)) for a in range(len(actions))] for j in range(bins)] for i in range(bins)])
    # w = np.random.rand((bins * bins * 3))
    model = defaultdict(tuple)

    steps_arr = []
    while True:
        predecessor = defaultdict(list)
        x_visited = []
        if iters == max_iters:
            break
        
        steps = 0
        start_state = (random.uniform(-0.6, -0.4), 0)
        s = start_state
        dis_s = discretize(s, bins)
        a = choose_action(q, dis_s, epsilon)
        while (dis_s[0] != bins - 1) and steps < 1000:
            x_visited.append(s[0])
            steps += 1
            next_state, reward = observe(s, a, bins)
            dis_next_state = discretize(next_state, bins)
            
            predecessor = update_leading_to_s(predecessor, dis_s, a, dis_next_state)
            
            model_update(model, dis_s, a, dis_next_state, reward)
            # print(model)
            priority = get_priority(q, dis_s, dis_next_state, reward, a)

            if priority > threshold:
                pq.put((-priority, dis_s, a))

            next_action = choose_action(q, dis_next_state, epsilon)
            a = next_action
            dis_s = dis_next_state
            s = next_state

            for itr_count in range(n_iters):
                if pq.empty():
                   break
                
                _, dis_s1, a1 = pq.get()
                r1, dis_s_prime1 = get_rs_from_model(model, dis_s1, a1)
                update_q(q, dis_s1, dis_s_prime1, a1, r1, alpha, gamma)

                for dis_s_, a_ in leading_to_s(predecessor,dis_s1):
                    r_, _ = get_rs_from_model(model, dis_s_, a_)
                    priority_ = get_priority(q, dis_s_, dis_s1, r_, a_)
                    
                    if priority_ > threshold:
                        pq.put((-priority_, dis_s_, a_))
            plt.plot(x_visited)
            plt.pause(0.0001)
        steps_arr.append(steps)
        iters += 1
        print(predecessor)
    return count
    



def get_pi(q, bins):
    pi = {}
    for i in range(bins):
        for j in range(bins):
            state = (i, j)
            x = q[state]
            a_star = np.argwhere(x == max(x)).flatten()
            new_pi_sa = []
            prob_a_star = ((1 - epsilon) / len(a_star)) + (epsilon / 3)
            prob_o = epsilon / 3
            for a in range(3):
                if a in a_star:
                    new_pi_sa.append(prob_a_star)
                else:
                    new_pi_sa.append(prob_o)
            pi[state] = [[0, 1, 2], new_pi_sa]
    return pi

def get_v(q, pi, bins):
    v = np.zeros((bins, bins))
    for i in range(bins):
        for j in range(bins):
            state = (i, j)
            probs = pi[state][1]
            q_s = q[state]
            for k in range(3):
                v[i][j] += probs[k] * q_s[k]
    return v


gamma = 0.90
alpha = 0.7
epsilon = 0.01
lam = 0.08
bins = 50
threshold = 0.0
n_iters = 5
max_iters = 20

q, iters, steps_arr = prio_sweep(threshold, n_iters, gamma, alpha, epsilon, max_iters, bins)
# print(steps_arr)
steps_reached = []
for i in steps_arr:
    if i < 1000:
        steps_reached.append(i)
print("Mean steps taken (during training): " + str(np.mean(steps_reached)))

eval = []
for i in range(10):
    s = (np.random.uniform(-0.6, -0.4), 0)
    ds = discretize(s, bins)
    count = 0
    while ds[0] != bins - 1 and count <= 1000:
        count += 1
        a = choose_action(q, ds, epsilon=0)
        s_prime, _ = observe(s, a, bins)
        ds_prime = discretize(s_prime, bins)
        s = s_prime
        ds = ds_prime
    eval.append(count)

for i in eval:
    if i > 1000:
        eval.remove(i)
print(eval)
print("Mean steps taken (during eval): " + str(np.mean(eval)))
# est_pi = get_pi(q, bins)
# print(get_v(q, est_pi, bins))

# x = np.zeros((5, 5, 4))
# x[3][2][0] = 1


# print(get_x(3, 2, 0))
# print(x.flatten())