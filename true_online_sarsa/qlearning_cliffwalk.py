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
    if next_state == (3, 11):
        reward = 0
    # if reward == -100:
        # print(s, a, next_state, reward)
    return next_state, reward
   

def sarsa(gamma, alpha, epsilon, max_iters, rows = 4, columns = 12):
    # ep -> s, a, r, s', a'
    iters = 0
    q = np.zeros((rows, columns, 4)) # q -> [["AU", "AR", "AD", "AL"]], 0 - AU 1 - AR, 2 - AD, 3 - AL 
    episodes_time = []
    while True:
        iters += 1
        # if (iters % 500) == 0 and epsilon > 0.10:
        #     epsilon -= 0.05
        #     print(epsilon)
        # start an episode
        s = random.sample(valid_states, 1)[0]
        episodes_time.append(iters)
        old_q = q.copy()
        while s != goal_state:
            # print(s)
            # print(q)
            a = choose_action(q, s, epsilon)
            next_state, reward = observe(s, a)
            next_action = np.argmax(q[next_state])
            q[s][a] += alpha * (reward + gamma * q[next_state][next_action] - q[s][a])
            episodes_time.append(iters)
            s = next_state
        
        # v_est = get_v(q, get_pi(q))
        # v_diff.append(np.sum(np.square(v_est - v_star))/23)
        # diff = np.abs(old_q - q)
        # if np.amax(diff) < 0.001:
            # break
        print("episode ended yaaay")
        if iters == max_iters:
            break

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

gamma = 0.90
alpha = 0.1
epsilon = 0.3

q, iters, to_plot = sarsa(gamma, alpha, epsilon, 1000)
pi = get_greedy_pi(q)
print_policy(pi)

'''
min_actions = 10000000
min_episodes = 10000000
a_ep_plots = []
v_diffs = []
qs = []
for i in range(20):
    print(i)
    q, iters, to_plot, v_diff = sarsa(gamma, alpha, epsilon)
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
print(avg_q)
pi = get_greedy_pi(avg_q)
print(pi)
print_policy(pi)
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