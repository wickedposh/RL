
import numpy as np

n_arms = 8
gamma = 0.1                      # exploration parameter, in (0,1]
base_pref = np.array([8,3,5,6,4,2,7,1]) / 8.0

from collections import deque
window = 5

def reward_sim(a,penalty):
    recent_count = list(history).count(a)
    if recent_count!=0:# how many of last `window` were arm a
        r = base_pref[a] * (penalty ** (recent_count))
    else:
        r=base_pref[a]# decay: over-played → lower reward
    history.append(a)                              # ← update AFTER computing reward
    return r    # already in [0,1] since base_pref ∈ [0,1] and penalty**k ≤ 1

def get_policy(w, gamma, K):
    p = w / w.sum()
    return (1 - gamma) * p + gamma / K      # mix with uniform → every prob ≥ gamma/K
def reward_int(a):
    score = float(input(f"Score for {choices[a]} (0-10): "))
    return score / 10.0    # normalize to [0,1] for EXP3

def exp3_step(w, gamma, K, reward_fn, penalty):
    pi = get_policy(w, gamma, K)             # 1. compute policy
    a = np.random.choice(K, p=pi)            # 2. SAMPLE an arm
    reward = reward_fn(a,penalty)                     # 3. observe reward (ONLY for a), in [0,1]
    r_hat = reward / pi[a]                    # 4. importance-weighted estimate  ← KEY
    w[a] *= np.exp(gamma * r_hat / K)         # 5. exponential-weights update     ← KEY
    return a, reward, pi
choices=["horror",'romcom','drama','scifi','anime','cartoon','fantasy','comedy']
decay_set=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
n_seeds=20
horror_probs=np.zeros((len(decay_set),n_seeds))
for i in range(len(decay_set)):
    for seed in range(n_seeds):
        plays = np.zeros(n_arms)
        w = np.ones(n_arms)  # weights, start uniform
        history = deque(maxlen=window)
        for t in range(1000):
            a, reward, pi = exp3_step(w, gamma, n_arms, reward_sim, decay_set[i])
            plays[a] += 1
        final_pi=get_policy(w,gamma,n_arms)
        horror_probs[i,seed]=final_pi[0]

mean=horror_probs.mean(axis=1)
std=horror_probs.std(axis=1)

import matplotlib.pyplot as plt
plt.errorbar(decay_set, mean, yerr=std, marker='o')
plt.xlabel("penalty (satiation retention)")
plt.ylabel("horror probability (concentration)")
plt.title("Satiation strength → policy concentration")
plt.show()
for i, penalty in enumerate(decay_set):
    print(f"penalty={penalty}: horror_prob = {mean[i]:.3f} ± {std[i]:.3f}")
