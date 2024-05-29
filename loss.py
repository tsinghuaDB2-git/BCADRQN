import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = open(r"./results/loss-data", "r").read()
    losses = data.split('\n')

    for i in losses:
        loss = i.split(',')
        r = []
        for j in loss:
            #if eval(j) > 1 and 1 > eval(loss[0]) > 0:
             #   continue
            r.append(eval(j))

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(np.arange(len(r)), r, 'r')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Cost')
        ax1.set_title('loss vs. training Epoch')
        plt.show()

