import json
import pandas as pd
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import product
import chineseize_matplotlib


config = json.load(open('config.json'))
AMOUNT = config['AMOUNT']
START_DATE = pd.to_datetime(config['START_DATE'])
END_DATE = pd.to_datetime(config['END_DATE'])
PERIOD_DATE = pd.to_datetime(config['PERIOD_DATE'])
BANK_INTEREST = config['BANK_INTEREST']
INTRA_INTEREST = config['INTRA_INTEREST']
MIN_OCCUPY_RATE = config['MIN_OCCUPY_RATE']

date = pd.read_csv('date.csv')
date['T'] = pd.to_datetime(date['T'])

first_day_after_period = date[date['T'] > PERIOD_DATE].iloc[0]['T']  # 2023-10-09
first_day_before_period = date[date['T'] <= PERIOD_DATE].iloc[-1]['T']  # 2023-09-28

# 如果全在09-26买，要求跨期有90%的资金的话，那占用率就是90%
# 如果全在09-28买，要求跨期有90%的资金的话，那占用率小于90%，大约是75.6%
min_occupy_rate = MIN_OCCUPY_RATE * \
    date_diff(first_day_before_period, first_day_after_period) / \
    date_diff(START_DATE, first_day_after_period)

print(date_span(START_DATE, first_day_after_period))
print(date_span(first_day_before_period, first_day_after_period))

# 使用二分法计算保本利率
def calculate_interest_inter_period(occupy_inter_period, occupy_intra_period):
    earn = AMOUNT * daily_interest(BANK_INTEREST - INTRA_INTEREST) * occupy_intra_period * date_span(first_day_after_period, END_DATE)
    l, r = 0, 100
    while r - l > 1e-10:
        m = (l + r) / 2
        cost = AMOUNT * daily_interest(m - BANK_INTEREST) * occupy_inter_period * date_diff(START_DATE, first_day_after_period)
        if earn > cost:  # 赚的多了，抬高利率
            l = m
        else:
            r = m
    return l

# 生成输入数据
x = np.linspace(min_occupy_rate, 1.0, 100)
y = np.linspace(0, 1.0, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i, j in product(range(X.shape[0]), range(X.shape[1])):
        occupy_inter_period = X[i, j]
        occupy_intra_period = Y[i, j]
        Z[i, j] = calculate_interest_inter_period(occupy_inter_period, occupy_intra_period)
print(Z)

# 绘制三维曲面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)

# # 设置坐标轴标签
ax.set_xlabel('跨期占用率')
ax.set_ylabel('期外占用率')
ax.set_zlabel('成本利率')

# 显示图形
plt.show()
