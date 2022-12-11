import numpy as np
from collections import defaultdict
import random
from matplotlib import pyplot as plt

states = []
for i in range(4):
    for j in range(12):
        states.append((i, j))

valid_states = states.copy()
cliff_states = []
start_state = (3, 0)
goal_state = (3, 11)

for j in range(1, 11):
    valid_states.remove((3, j))
    cliff_states.append((3, j))

actions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
actions_rev = {0: "AU", 1: "AR", 2: "AD", 3: "AL"} 

def choose_action(q, s, epsilon):
    x = np.array(q[s])
    a_star = list(np.argwhere(x == max(x)).flatten())
    prob = np.random.uniform()
    if prob < epsilon:
        return random.sample([0, 1, 2, 3], 1)[0]
    else:
        return random.sample(a_star, 1)[0]

def observe(s, a):
    dir = actions[a]
    next_state = (s[0]+dir[0], s[1]+dir[1])
    reward = -1
    # hits boundary, stays in the same state
    if next_state[0] < 0 or next_state[0] > 3 or next_state[1] < 0 or next_state[1] > 11:
        next_state = s
    # falls into cliff, goes back to start state, reward = -100
    if next_state in cliff_states:
        next_state = start_state
        reward = -100
    # if reward == -100:
        # print(s, a, next_state, reward)
    return next_state, reward

def get_x(s, a, rows = 4, columns = 12, actions = 4):
    x = np.zeros((rows * columns * actions))
    index = s[0]*(columns * actions) + s[1]*actions + a
    if s != goal_state:
        x[index] = 1
    return x

def true_online_sarsa(gamma, alpha, lam, epsilon, max_iters, rows = 4, columns = 12, actions = 4):
    iters = 1
    q = np.zeros((rows, columns, actions)) # q -> [["AU", "AR", "AD", "AL"]], 0 - AU 1 - AR, 2 - AD, 3 - AL 
    w = np.random.rand((rows * columns * actions))
    episodes_time = []
    while True:
        if iters == max_iters:
            break
        # if (iters % 2000) == 0 and epsilon > 0.10:
        #     epsilon -= 0.05
        #     print(epsilon)
        # epsilon = min(epsilon, 20*epsilon/iters)
        # start an episode
        s = random.sample(valid_states, 1)[0]
        a = choose_action(q, s, epsilon)
        x = get_x(s, a)
        Q_old = 0
        z = np.zeros((rows * columns * actions))
        episodes_time.append(iters)
        while s != goal_state:
            # print(s)
            episodes_time.append(iters)
            next_state, reward = observe(s, a)
            next_action = choose_action(q, next_state, epsilon)
            next_x = get_x(next_state, next_action)
            # print("x_shape: ")
            # print(x.shape)
            # print("w shape: ")
            # print(w.shape)
            Q = np.dot(w, x)
            next_Q = np.dot(w, next_x)
            # print("Q': " + str(next_Q))
            q[next_state[0]][next_state[1]][next_action] = next_Q 
            delta = reward + gamma * next_Q - Q
            z = gamma * lam * z + (1 - alpha * gamma * lam * (z.T) * x) * x
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = next_Q
            x = next_x
            a = next_action
            s = next_state
            # print(q)
            # q[s[0]][s[1]][a] += alpha * (reward + gamma * q[next_state[0]][next_state[1]][next_action] - q[s[0]][s[1]][a])
        # print("goal state reached yaay!")
        iters += 1
        

    return q, iters, episodes_time

def get_greedy_pi(q):
    pi = {}
    for state in valid_states:
        pi[state] = np.argmax(q[state])
    return pi

def get_pi(q, rows = 4, columns = 12):
    pi = {}
    for i in range(rows):
        for j in range(columns):
            state = (i, j)
            x = q[state]
            a_star = np.argwhere(x == max(x)).flatten()
            new_pi_sa = []
            prob_a_star = ((1 - epsilon) / len(a_star)) + (epsilon / 4)
            prob_o = epsilon / 4
            for a in range(4):
                if a in a_star:
                    new_pi_sa.append(prob_a_star)
                else:
                    new_pi_sa.append(prob_o)
            pi[state] = [["AU", "AR", "AD", "AL"], new_pi_sa]
    return pi

def get_v(q, pi, rows = 4, columns = 12):
    v = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            state = (i, j)
            probs = pi[state][1]
            q_s = q[state]
            for k in range(4):
                v[i][j] += probs[k] * q_s[k]
    return v

def print_v(v):
    (rows, cols) = v.shape
    for i in range(rows):
        print("\t".join([str(round(v[i][j], 4)) for j in range(cols)]))

def print_policy(pi, rows = 4, columns = 12):
    # {0: "AU", 1: "AR", 2: "AD", 3: "AL"} 
    for i in range(rows):
        l = []
        for j in range(columns):
            state = (i, j)
            if state == goal_state:
                l.append('G')
            elif state in cliff_states:
                l.append(' ')
            elif pi[state] == 0:
                l.append("\u2191")
            elif pi[state] == 1:
                l.append("\u2192")
            elif pi[state] == 2:
                l.append("\u2193")
            else:
                l.append("\u2190")
        print("\t".join(l))


# gamma = 0.90
# alpha = 0.1
# epsilon = 0.1
# lam = 0.1

# q, iters = true_online_sarsa(gamma, alpha, lam, epsilon, v_star, 1000)
# est_pi = get_greedy_pi(q)
# print_policy(est_pi)
# print_v(get_v(q, get_pi(q)))

gamma = 0.90
alpha = 0.05
epsilon = 0.1
lam = 0.05


q, iters, to_plot = true_online_sarsa(gamma, alpha, lam, epsilon, 10000)
pi = get_greedy_pi(q)
print_policy(pi)
print(get_pi(q))

# gamma = 0.90
# alpha = 0.05
# epsilon = 0.1
# lam = 0.08
'''
min_actions = 10000000
min_episodes = 10000000
a_ep_plots = []
v_diffs = []
qs = []
for i in range(1):
    print(i)
    q, iters, to_plot = true_online_sarsa(gamma, alpha, lam, epsilon, 2500)
    print(iters)
    qs.append(q)
    min_actions = min(min_actions, len(to_plot))
    a_ep_plots.append(to_plot)
    min_episodes = min(min_episodes, len(v_diff))
    v_diffs.append(v_diff)

plot_data_1 = []
plot_data_2 = []

for plot in a_ep_plots:
    plot_data_1.append(plot[:min_actions])

for plot in v_diffs:
    plot_data_2.append(plot[:min_episodes])

avg_q = np.mean(qs, axis = 0)
pi = get_greedy_pi(avg_q)
print_policy(pi)
v_est = get_v(avg_q, get_pi(avg_q))
print("Estimated V: ")
print_v(v_est)
plt.figure(0)
plt.plot(np.mean(np.array(plot_data_1), axis = 0))
plt.xlabel("Number of actions")
plt.ylabel("Number of episodes")
plt.figure(1)
plt.plot(np.mean(np.array(plot_data_2), axis = 0))
plt.xlabel("Number of episodes")
plt.ylabel("MSE")
plt.show()
'''