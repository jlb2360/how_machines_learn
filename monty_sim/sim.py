import numpy as np
import random
import matplotlib.pyplot as plt


iters = 1000
prob_stay = np.zeros(iters)
prob_flip = np.zeros(iters)
flip_correct = 0
stay_correct = 0


for i in range(iters):
    true_idx = random.randint(0, 2)

    # initial pick
    pick = random.randint(0, 2)

    if pick != true_idx:
        flip_correct += 1
    else:
        stay_correct += 1

    prob_stay[i] =  stay_correct/(i+1)
    prob_flip[i] = flip_correct/(i+1)

plt.plot(prob_stay, label="probability staying wins")
plt.plot(prob_flip, label="probability fliping wins")
plt.legend()
plt.show()





        