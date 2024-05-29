import numpy as np
import csv
import math
from tqdm import tqdm

class Buf:
    def __init__(self):
        self.buffer = []
        self.data = []
        self.size = 0
        self.temp = 0

    def create(self):
        print('Memory creating...')
        csv_file = csv.reader(open(f'./data/final.csv', 'r'))
        content = []
        for line in csv_file:
            content.append(line)

        data = content[1:]

        for i in tqdm(range(len(data)-1)):

            buf = []
            s1 = []
            s2 = []
            action = []
            reward = []

            ID = eval(data[i][0])
            for j in data[i][1:40]:
                s1.append(eval(j))
            for j in data[i][40:79]:
                s2.append(eval(j))
            # for j in data[i][79:93]:
            for j in data[i][83:84]:
                action.append(eval(j))
            for j in data[i][86:90]:
                if eval(j) > 2:
                    j = '2'
                if eval(j) < -2:
                    j = '-2'
                action.append(eval(j))


            # reward function desgin
            r = (-50) * (eval(data[i][92]) - eval(data[i][90])) + (-10) * (eval(data[i][93]) - eval(data[i][91]))

            if eval(data[i][92]) == eval(data[i][90]) and eval(data[i][92]) > 0:
                r -= 5 * eval(data[i][92])

            if 0 < r < 1:
                r = 1
            elif -1 < r <= 0:
                r = -1

            if data[i][0] != data[i + 1][0]:
                if eval(data[i][-1]) == 1:
                    r = -100
                if eval(data[i][-1]) == 0:
                    r = 100

            reward.append(r)

            buf.append(ID)
            buf.append(s1)
            buf.append(s2)
            buf.append(action)
            buf.append(reward)

            self.buffer.append(buf)
            self.size += 1

        temp = []
        i = 0
        while i < len(self.buffer) - 1:
            if self.buffer[i][0] == self.buffer[i+1][0]:
                temp.append(self.buffer[i])
                i += 1
            else:
                temp.append(self.buffer[i])
                self.data.append(temp)
                i += 1
                temp = []


    def step(self):
        i = self.temp
        states = np.array(self.buffer[i][1], copy=False)
        next_states = np.array(self.buffer[i][2], copy=False)
        actions = np.array(self.buffer[i][3], copy=False)
        rewards = np.array(self.buffer[i][4], copy=False)
        self.temp += 1

        return states, next_states, actions, rewards

    def check(self):
        i = self.temp - 1
        if i == self.size - 1:
            return True
        else:
            if self.buffer[i][0] == self.buffer[i+1][0]:
                return False
            else:
                return True


    def getEpisode(self):
        i = np.random.randint(0, len(self.data))

        return self.data[i]

    def getMaxAction(self):
        # l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        l = [0, 0, 0, 0, 0]
        for i in range(len(self.buffer)):
            for j in range(len(l)):
                if self.buffer[i][3][j] >= l[j]:
                    l[j] = self.buffer[i][3][j]
                    if l[j] > 5:
                        l[j] = 5

        return np.array(l)

# test
if __name__ == '__main__':
    buffer = Buf()
    buffer.create()
    episode = buffer.getEpisode()
    print(buffer.getMaxAction())
