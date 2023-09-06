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
x = np.linspace(min_occupy_rate, 1.0, 101)
y = np.linspace(0, 1.0, 101)
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
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(3, 2, 1, projection='3d')
ax.plot_surface(X, Y, Z)
ax.set_xlabel('跨期占用率')
ax.set_ylabel('非跨期占用率')
ax.set_zlabel('成本利率')

# 绘制 x=1.0 的截面曲面图
ax = fig.add_subplot(3, 2, 3)
twoD = np.array([(y, z) for (x, y, z) in points if x == 0.9])
y, z = zip(*twoD)
ax.plot(y, z, label='跨期占用率0.9')
twoD = np.array([(y, z) for (x, y, z) in points if x == 1.0])
y, z = zip(*twoD)
ax.plot(y, z, label='跨期占用率1.0')
ax.legend()
ax.set_xlabel('非跨期占用率')
ax.set_ylabel('成本利率')

# 绘制 x=0.9 的截面曲面图
ax = fig.add_subplot(3, 2, 4)
twoD = np.array([(x, z) for (x, y, z) in points if y == 0.33])
x, z = zip(*twoD)
ax.plot(y, z, label='非跨期占用率0.33')
twoD = np.array([(x, z) for (x, y, z) in points if y == 0.67])
x, z = zip(*twoD)
ax.plot(y, z, label='非跨期占用率0.67')
twoD = np.array([(x, z) for (x, y, z) in points if y == 1.00])
x, z = zip(*twoD)
ax.plot(y, z, label='非跨期占用率1.00')
ax.legend()
ax.set_xlabel('跨期占用率')
ax.set_ylabel('成本利率')

# 绘制 x=0.9 的截面曲面图
ax = fig.add_subplot(3, 2, 5)
scale = AMOUNT // 100000000
twoD = np.array([(y * scale, z) for (x, y, z) in points if x == 0.9])
y, z = zip(*twoD)
ax.plot(y, z, label=f'跨期日均权益{0.9 * scale:.2f}亿')
twoD = np.array([(y * scale, z) for (x, y, z) in points if x == 1.0])
y, z = zip(*twoD)
ax.plot(y, z, label=f'跨期日均权益{1.0 * scale:.2f}亿')
ax.legend()
ax.set_xlabel('非跨期日均权益(亿元)')
ax.set_ylabel('成本利率')

# 绘制 x=0.9 的截面曲面图
ax = fig.add_subplot(3, 2, 6)
scale = AMOUNT // 100000000
twoD = np.array([(x * scale, z) for (x, y, z) in points if y * 100 // 1 == 33])
x, z = zip(*twoD)
ax.plot(x, z, label=f'非跨期日均权益{0.33 * scale:.2f}亿')
twoD = np.array([(x * scale, z) for (x, y, z) in points if y * 100 // 1 == 67])
x, z = zip(*twoD)
ax.plot(x, z, label=f'非跨期日均权益{0.67 * scale:.2f}亿')
twoD = np.array([(x * scale, z) for (x, y, z) in points if y == 1.00])
x, z = zip(*twoD)
ax.plot(x, z, label=f'非跨期日均权益{1.00 * scale:.2f}亿')
ax.legend()
ax.set_xlabel('跨期日均权益(亿元)')
ax.set_ylabel('成本利率')

plt.savefig('result.png')


# 生成输入数据
x = np.linspace(0, 1.0, 101)
y = np.linspace(0, 1.0, 101)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
points = []
for i, j in product(range(X.shape[0]), range(X.shape[1])):
    occupy_inter_period = X[i, j]
    occupy_intra_period = Y[i, j]
    Z[i, j] = calculate_interest_inter_period(occupy_inter_period, occupy_intra_period)
    points.append((occupy_inter_period, occupy_intra_period, Z[i, j]))

scale = AMOUNT // 100000000
columns = [f"{num}亿" for num in range(10, 0, -1)]
index = [f"{num}亿" for num in range(10, 0, -1)]
df = pd.DataFrame(columns=columns, index=index)
for i, j in product(range(10, 0, -1), range(10, 0, -1)):
    df.loc[f"{i}亿", f"{j}亿"] = Z[i*10, j*10] * 100 // 1 / 100 # 这里不优雅，有时间再改吧
df.to_csv('first.csv')

df = []
for interest in range(324, 331):
    interest /= 100
    earn = AMOUNT * daily_interest(BANK_INTEREST - INTRA_INTEREST) * 1.0 * len(intra_day)
    cost = AMOUNT * daily_interest(interest - BANK_INTEREST) * 0.9 * len(inter_day)
    df.append((interest, (earn - cost) / 10000))
df = pd.DataFrame(df, columns=['跨期返息率', '最终盈亏(万元)'])
df.to_csv('second.csv')