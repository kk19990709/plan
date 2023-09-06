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
date['expire_T'] = pd.to_datetime(date['expire_T'])
date = date[(date['T'] >= START_DATE) & (date['expire_T'] <= END_DATE)]
# 因为周末无法买入，9.18开始的话，最晚10.13(周五)买入，根据双休日做调整可以显著增大收益
# 如果9.21开始，最晚10.18买入
date.to_csv('date')

T_before_period = date[date['T'] <= PERIOD_DATE].iloc[-1]['T']  # 2023-09-28
T_after_period = date[date['T'] > PERIOD_DATE].iloc[0]['T']  # 2023-10-09

intra_T = date[date['expire_T'] != T_after_period]
inter_T = date[date['expire_T'] == T_after_period]
# 跨期的存款和非跨期的存款会有一些重叠。比如9.25买的非跨期存款9.28可以取出，那么9.26和9.27会和跨期存款有一些重叠
# 我们需要尽可能地在9.28多拿跨期订单，9.26和9.27少拿跨期订单
intra_day = set()
for T in intra_T.itertuples():
    # 我们强制要求非跨期的必须在9.28前取走，不算9.28的收益。也就是10月前最晚9.25买入，算9.25+9.26+9.27的收益
    intra_day.update(pd.date_range(T.T, T.expire_T, freq='D', inclusive='both'))
intra_day.discard(T_before_period)
inter_day = set()
for T in inter_T.itertuples():
    if T.T < T_before_period: continue  # 这里我们强制要求跨期全在9.28拿
    inter_day.update(pd.date_range(T.T, T.expire_T, freq='D', inclusive='left'))
intra_day = sorted(list(map(lambda x: x.strftime('%Y-%m-%d'), intra_day)))
inter_day = sorted(list(map(lambda x: x.strftime('%Y-%m-%d'), inter_day)))

# ['2023-09-28', '2023-09-29', '2023-09-30', '2023-10-01', '2023-10-02', '2023-10-03', '2023-10-04', '2023-10-05', '2023-10-06', '2023-10-07', '2023-10-08'] 11
print(inter_day, len(inter_day))
# ['2023-09-18', '2023-09-19', '2023-09-20', '2023-09-21', '2023-09-22', '2023-09-23', '2023-09-24', '2023-09-25', '2023-09-26', '2023-09-27', '2023-10-09', '2023-10-10', '2023-10-11', '2023-10-12', '2023-10-13', '2023-10-14', '2023-10-15'] 17
print(intra_day, len(intra_day))

# 如果全在09-26买，要求跨期有90%的资金的话，那占用率就是90%
# 如果全在09-28买，要求跨期有90%的资金的话，那占用率小于90%，大约是75.6%
min_occupy_rate = MIN_OCCUPY_RATE  # 因为我们强制要求买入时间了，所以这里就是90%

# 使用二分法计算保本利率
def calculate_interest_inter_period(occupy_inter_period, occupy_intra_period):
    earn = AMOUNT * daily_interest(BANK_INTEREST - INTRA_INTEREST) * occupy_intra_period * len(intra_day)
    l, r = 0, 100
    while r - l > 1e-10:
        m = (l + r) / 2
        cost = AMOUNT * daily_interest(m - BANK_INTEREST) * occupy_inter_period * len(inter_day)
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
points = []
for i, j in product(range(X.shape[0]), range(X.shape[1])):
    occupy_inter_period = X[i, j]
    occupy_intra_period = Y[i, j]
    Z[i, j] = calculate_interest_inter_period(occupy_inter_period, occupy_intra_period)
    points.append((occupy_inter_period, occupy_intra_period, Z[i, j]))
print(Z)

# 绘制三维曲面
fig = plt.figure()
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(X, Y, Z)

# # 设置坐标轴标签
ax.set_xlabel('跨期占用率')
ax.set_ylabel('非跨期占用率')
ax.set_zlabel('成本利率')

# 绘制 x=1.0 的截面曲面图
ax2 = fig.add_subplot(2, 2, 2)
twoD = np.array([(y, z) for (x, y, z) in points if x == 0.9])
y, z = zip(*twoD)
ax2.plot(y, z, label='跨期占用率0.9')
ax2 = fig.add_subplot(2, 2, 2)
twoD = np.array([(y, z) for (x, y, z) in points if x == 1.0])
y, z = zip(*twoD)
ax2.plot(y, z, label='跨期占用率1.0')
ax2.legend()

# 设置坐标轴标签
ax2.set_xlabel('非跨期占用率')
ax2.set_ylabel('成本利率')

# 绘制 x=0.9 的截面曲面图
ax3 = fig.add_subplot(2, 2, 3)
twoD = np.array([(x, z) for (x, y, z) in points if y * 100 // 1 == 33])
x, z = zip(*twoD)
ax3.plot(y, z, label='非跨期占用率0.33')
twoD = np.array([(x, z) for (x, y, z) in points if y * 100 // 1 == 67])
x, z = zip(*twoD)
ax3.plot(y, z, label='非跨期占用率0.67')
twoD = np.array([(x, z) for (x, y, z) in points if y == 1.00])
x, z = zip(*twoD)
ax3.plot(y, z, label='非跨期占用率1.00')
ax3.legend()

# 设置坐标轴标签
ax3.set_xlabel('跨期占用率')
ax3.set_ylabel('成本利率')

plt.show()
