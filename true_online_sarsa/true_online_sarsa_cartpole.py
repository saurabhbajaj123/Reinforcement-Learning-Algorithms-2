import numpy as np
from collections import defaultdict
import random
from matplotlib import pyplot as plt
import math
import gym



def discretize(state, bins):
    (x, v, theta, theta_v) = state
    x_index = (x - (-4.8))/(4.8-(-4.8))*bins
    theta_index = (theta - (-0.418))/(0.418-(-0.418))*bins
    v_index = (v - (-5))/(5-(-5))*bins
    theta_v_index = (theta_v - (-5))/(5-(-5))*bins
    if x_index == bins:
        x_index = bins-1
    if theta_index == bins:
        theta_index = bins - 1
    if v_index == bins:
        v_index = bins-1
    if theta_v_index == bins:
        theta_v_index = bins - 1
    return (math.floor(x_index), math.floor(v_index), math.floor(theta_index), math.floor(theta_v_index))

# actions -> [0, 1] = [0, 1]
def choose_action(q, s, epsilon):
    x = np.array(q[s])
    a_star = list(np.argwhere(x == max(x)).flatten())
    prob = np.random.uniform()
    if prob < epsilon:
        return random.sample([0, 1], 1)[0]
    else:
        return random.sample(a_star, 1)[0]

# def observe(s, a, no_of_bins):
#     x_t, v_t = s
#     next_v = v_t + 0.001 * actions[a] - 0.0025 * math.cos(3 * x_t)
#     next_x = x_t + next_v

#     if next_v < -0.07:
#         next_v = -0.07
#     if next_v > 0.07:
#         next_v = 0.07

#     if next_x < -1.2:
#         next_x = -1.2
#         next_v = 0
#     if next_x > 0.5:
#         next_x = 0.5
#         next_v = 0

#     reward = -1
#     if discretize((next_x, next_v), no_of_bins)[0] == no_of_bins-1:
#         reward = 0
    
#     return (next_x, next_v), reward

def get_x(state, action, bins):
    x = np.zeros((bins,bins,bins,bins,2)) 
    if np.abs(state[0]) > 2.4 or np.abs(state[2]) > 0.2095:
        return x.flatten()
    dstate = discretize(state, bins)
    x[dstate][action] = 1
    return x.flatten()

def true_online_sarsa(gamma, alpha, lam, epsilon, max_iters, bins):
    iters = 0
    q = np.zeros((bins, bins, bins, bins, 2)) # q -> [["0", "1"]], 0 - -1, 1 - 0, 2 - 1
    w = np.random.rand((bins * bins * bins * bins * 2))
    episodes_time = []
    steps_arr = []
    while True:
        print(iters)
        if iters == max_iters:
            break
        if (iters % 500) == 0 and alpha > 0.05:
            alpha -= 0.05
            # print(alpha)
        if (iters % 500) == 0 and epsilon > 0.1:
            epsilon -= 0.05
        # start an episode
        steps = 0
        start_state = env.reset()[0]
        # print(start_state)
        # start_state = (random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05))
        s = start_state
        dis_s = discretize(s, bins)
        a = choose_action(q, dis_s, epsilon)
        x = get_x(s, a, bins)
        Q_old = 0
        z = np.zeros((bins * bins * bins * bins * 2))
        episodes_time.append(iters)
        terminated = False
        truncated = False
        while True:
            episodes_time.append(iters)
            steps += 1
            next_state, reward, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
            dis_next_state = discretize(next_state, bins)
            next_action = choose_action(q, dis_next_state, epsilon)
            next_x = get_x(next_state, next_action, bins)
            # print("x_shape: ")
            # print(x.shape)
            # print("w shape: ")
            # print(w.shape)
            Q = np.dot(w, x)
            next_Q = np.dot(w, next_x)
            # print("Q': " + str(next_Q))
            # q[dis_next_state][next_action] = next_Q 
            delta = reward + gamma * next_Q - Q
            z = gamma * lam * z + (1 - alpha * gamma * lam * (z.T) * x) * x
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            q[dis_s][a] = np.dot(w, x)
            q[dis_next_state][next_action] = np.dot(w, next_x)
            Q_old = next_Q
            x = next_x
            s = next_state
            dis_s = dis_next_state
            a = next_action

        steps_arr.append(steps)
        iters += 1
        

    return q, iters, episodes_time, steps_arr

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

def evaluate_policy(q, bins):
    eval = []
    for i in range(10):
        s = env.reset()[0]
        ds = discretize(s, bins)
        count = 0
        terminated = False
        truncated = False
        while True:
            if terminated or truncated:
                break
            count += 1
            a = choose_action(q, ds, epsilon=0)
            s_prime, reward, terminated, truncated, info = env.step(a)
            ds_prime = discretize(s_prime, bins)
            s = s_prime
            ds = ds_prime
        eval.append(count)

    return eval

# gamma = 0.90
# alpha = 0.05
# epsilon = 0.1
# lam = 0.08
# bins = 15

# q, iters, steps_arr = true_online_sarsa(gamma, alpha, lam, epsilon, 2000, bins)
# # print(steps_arr)
# steps_reached = []
# for i in steps_arr:
#     if i < 1000:
#         steps_reached.append(i)
# print("Mean steps taken (during training): " + str(np.mean(steps_reached)))


gamma = 1.0
alpha = 0.15
epsilon = 0.5
lam = 0.08
bins = 20
env = gym.make('CartPole-v1')

min_actions = 10000000
min_episodes = 1000
a_ep_plots = []
steps_arrs = []
qs = []
max_steps = 0
for i in range(2):
    print(i)
    q, iters, to_plot, steps_arr = true_online_sarsa(gamma, alpha, lam, epsilon, 1000, bins)
    qs.append(q)
    max_steps = max(max_steps, max(steps_arr))
    min_actions = min(min_actions, len(to_plot))
    a_ep_plots.append(to_plot)
    min_episodes = min(min_episodes, iters)
    steps_arrs.append(steps_arr)

plot_data_1 = []
plot_data_2 = []

for plot in a_ep_plots:
    plot_data_1.append(plot[:min_actions])

for plot in steps_arrs:
    plot_data_2.append(plot[:min_episodes])

avg_q = np.mean(qs, axis = 0)
eval = evaluate_policy(avg_q, bins)
print("Mean steps taken (during eval): " + str(np.mean(eval)))
print(max_steps)

plt.figure(0)
plt.plot(np.mean(np.array(plot_data_1), axis = 0))
plt.xlabel("Number of actions")
plt.ylabel("Number of episodes")
plt.figure(1)
plt.errorbar(np.arange(1000)[::5], np.mean(plot_data_2, axis = 0)[::5],yerr=np.std(plot_data_2, axis = 0)[::5], fmt = '.', capsize=1)
plt.xlabel("Number of episodes")
plt.ylabel("Number of steps needed to reach goal state")
plt.show()
