import numpy as np
import matplotlib.pyplot as plt
import csv

def evaluate_BCQ(step=0):

    # files needs to be changed
    csv_file = csv.reader(open(f'./results/final-0.01.csv', 'r', encoding='UTF-8'))
    content = []
    for line in csv_file:
        content.append(line)

    # 按照Q来评估
    data = content[1:200000:5]
    Q, new_Q = 0, 0
    for item in data:
        Q += eval(item[-8])
        new_Q += eval(item[-2])
    Q = Q / len(data)
    new_Q = new_Q / len(data)

    # 计算原始的死亡率
    num = 0
    mor = 0
    for item in data:
        if Q - 5 < eval(item[-8]) < Q + 5:
            num += 1
            if eval(item[-1]) == 0:
                mor += 1
    rate1 = 1 - mor / num
    print('原始策略的平均Q值为' + str(Q))
    print('原始策略的生存率为' + str(rate1) + '\n')


    # 计算模型的死亡率
    num = 0
    mor = 0
    for item in data:
        if new_Q - 5 < eval(item[-2]) < new_Q + 5:
            num += 1
            if eval(item[-1]) == 0:
                mor += 1
    rate2 = 1 - mor / num
    print('模型策略的平均Q值为' + str(new_Q))
    print('模型策略的生存率为' + str(rate2))
    print()

    return Q, rate2

if __name__ == "__main__":
    Q, S = evaluate_BCQ(0.001)