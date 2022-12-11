import numpy as np
from collections import defaultdict
import random
from matplotlib import pyplot as plt


valid_states = [(0,0), (0,1), (0,2), (0,3), (0,4), (1,0), (1,1), (1,2), (1,3), (1,4), (2,0), (2,1), (2,3), (2,4), (3,0), (3,1), (3,3), (3,4), (4,0), (4,1), (4,2), (4,3)]
actions = { 3: [(-1, 0), (0, -1), (0, 1)], 0: [(0, 1), (1, 0), (-1, 0)], 1: [(0, -1), (1, 0), (-1, 0)], 2: [(1, 0), (0, -1), (0, 1)] }
actions_rev = {0: "AR", 1: "AL", 2: "AD", 3: "AU"} 

def choose_action(q, s, epsilon):
    x = np.array(q[s])
    a_star = list(np.argwhere(x == max(x)).flatten())
    prob = np.random.uniform()
    if prob < epsilon:
        return random.sample([0, 1, 2, 3], 1)[0]
    else:
        return random.sample(a_star, 1)[0]

def observe(s, a):
    a_prob = np.random.uniform()
    dir = None
    if a_prob <= 0.8:
        dir = actions[a][0]
    elif a_prob <= 0.85:
        dir = actions[a][1]
    elif a_prob <= 0.90: 
        dir = actions[a][2]
    else:
        dir = (0, 0)

    next_state = (s[0]+dir[0], s[1]+dir[1])
    if next_state[0] < 0 or next_state[0] > 4 or next_state[1] < 0 or next_state[1] > 4 or next_state == (2, 2) or next_state == (3, 2):
        next_state = s

    reward = 0
    if next_state == (4, 4):
        reward = 10
    if next_state == (4, 2):
        reward = -10
    
    return next_state, reward

def get_x(state, action):
    x = np.zeros((100))
    index = state[0]*20 + state[1]*4 + action
    if state != (4, 4):
        x[index] = 1
    return x

def true_online_sarsa(gamma, alpha, lam, epsilon, v_star, max_iters):
    iters = 1
    q = np.zeros((5, 5, 4)) # q -> [["AR", "AL", "AD", "AU"]], 0 - AR, 1 - AL, 2 - AD, 3 - AU
    w = np.random.rand((100))
    episodes_time = []
    mses = []
    max_norms = []
    while True:
        if iters == max_iters:
            break
        if (iters % 1500) == 0 and alpha > 0.05:
            alpha -= 0.05
            # print(alpha)
        if (iters % 500) == 0 and epsilon > 0.1:
            epsilon -= 0.05
            # print(epsilon)
        # epsilon = min(epsilon, 20 * epsilon/iters)
        # print(epsilon)
        # alpha = min(alpha, 20 * alpha/iters)
        # start an episode
        start_state = random.sample(valid_states, 1)[0]
        s = start_state
        a = choose_action(q, s, epsilon)
        x = get_x(s, a)
        Q_old = 0
        z = np.zeros((100))
        episodes_time.append(iters)
        while s != (4, 4):
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

        v_est = get_v(q, get_pi(q))
        mses.append(np.square(np.subtract(v_est, v_star)).mean())
        max_norms.append((np.amax(np.abs(v_est - v_star))))
        # if np.amax(v_diff) < 0.001:
        #     break
        iters += 1
        

    return q, iters, episodes_time, mses, max_norms

def get_greedy_pi(q):
    pi = {}
    for state in valid_states:
        pi[state] = np.argmax(q[state])
    return pi

def get_pi(q):
    pi = {}
    for i in range(5):
        for j in range(5):
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
            pi[state] = [["AR", "AL", "AD", "AU"], new_pi_sa]
    return pi

def get_v(q, pi):
    v = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
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

def print_policy(pi):
    #{0: "AR", 1: "AL", 2: "AD", 3: "AU"} 
    for i in range(5):
        l = []
        for j in range(5):
            state = (i, j)
            if (i == 4 and j == 4):
                l.append('G')
            elif (i == 2 and j == 2):
                l.append(' ')
            elif (i == 3 and j == 2):
                l.append(' ')
            elif pi[state] == 3:
                l.append("\u2191")
            elif pi[state] == 0:
                l.append("\u2192")
            elif pi[state] == 2:
                l.append("\u2193")
            else:
                l.append("\u2190")
        print("\t".join(l))

v_star = np.array([
    [4.0187, 4.5548, 5.1575, 5.8336, 6.4553],
    [4.3716, 5.0324, 5.8013, 6.6473, 7.3907],
    [3.8672, 4.3900, 0.0000, 7.5769, 8.4637],
    [3.4182, 3.8319, 0.0000, 8.5738, 9.6946],
    [2.9977, 2.9309, 6.0733, 9.6946, 0.0000]
])

pi_star = {
    (0,0):'AR', (0,1):'AR', (0,2):'AR', (0,3):'AD', (0,4):'AD', 
    (1,0):'AR', (1,1):'AR', (1,2):'AR', (1,3):'AD', (1,4):'AD', 
    (2,0):'AU', (2,1):'AU',             (2,3):'AD', (2,4):'AD',
    (3,0):'AU', (3,1):'AU',             (3,3):'AD', (3,4):'AD', 
    (4,0):'AU', (4,1):'AU', (4,2):'AR', (4,3):'AR', (4,4):'G'
}


# gamma = 0.90
# alpha = 0.1
# epsilon = 0.1
# lam = 0.1

# q, iters = true_online_sarsa(gamma, alpha, lam, epsilon, v_star, 1000)
# est_pi = get_greedy_pi(q)
# print_policy(est_pi)
# print_v(get_v(q, get_pi(q)))

# gamma = 0.90
# alpha = 0.05
# epsilon = 0.3
# lam = 0.1

# 2500 episodes, epsilon = 0.1, gamma = 0.9, alpha = 0.1, lamda = 0.1
gamma = 0.90
alpha = 0.2
epsilon = 0.3
lam = 0.08

min_actions = 10000000
min_episodes = 10000000
a_ep_plots = []
mses = []
max_norms = []
qs = []
for i in range(20):
    print(i)
    q, iters, to_plot, mse, max_norm = true_online_sarsa(gamma, alpha, lam, epsilon, v_star, 3000)
    print(iters)
    qs.append(q)
    min_actions = min(min_actions, len(to_plot))
    a_ep_plots.append(to_plot)
    min_episodes = min(min_episodes, len(mse))
    mses.append(mse)
    max_norms.append(max_norm)

plot_data_1 = []
plot_data_2 = []
plot_data_3 = []

for plot in a_ep_plots:
    plot_data_1.append(plot[:min_actions])

for plot in mses:
    plot_data_2.append(plot[:min_episodes])

for plot in max_norms:
    plot_data_3.append(plot[:min_episodes])

avg_q = np.mean(qs, axis = 0)
pi = get_greedy_pi(avg_q)
print_policy(pi)
v_est = get_v(avg_q, get_pi(avg_q))
print("V_star: ")
print_v(v_star)
print("Estimated V: ")
print_v(v_est)
print("MSE: ")
print(np.square(np.subtract(v_est, v_star)).mean())
print("Max Norm: ")
print(np.amax(np.abs(v_est - v_star)))
plt.figure(0)
plt.plot(np.mean(np.array(plot_data_1), axis = 0))
plt.xlabel("Number of actions")
plt.ylabel("Number of episodes")
plt.figure(1)
plt.plot(np.mean(np.array(plot_data_2), axis = 0))
plt.xlabel("Number of episodes")
plt.ylabel("MSE")
plt.figure(2)
plt.plot(np.mean(np.array(plot_data_3), axis = 0))
plt.xlabel("Number of episodes")
plt.ylabel("Max Norm")
plt.show()
