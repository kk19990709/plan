import pandas as pd
import json

holiday = open('holiday.txt', encoding='utf-8').readlines()
holiday = [x.strip() for x in holiday]
holiday = set(holiday)

config = json.load(open('config.json'))

def trade_date(start, end):
    dates = []
    for date in pd.date_range(start=start, end=end, freq='D'):
        if date.weekday() not in [5, 6] and date.strftime("%Y-%m-%d") not in holiday:
            dates.append(date.strftime("%Y-%m-%d"))
    return dates

df = pd.DataFrame(trade_date('2023-09-01', '2023-12-31'), columns=['T'])
df['T'] = pd.to_datetime(df['T'])
delta = pd.Timedelta(days=config['SPAN'])
df['expire_T'] = df['T'].apply(lambda x: df['T'].searchsorted(x+delta))
df['expire_T'] = df['expire_T'].apply(lambda x: df.iloc[x]['T'] if x < len(df) else pd.NaT)
df.to_csv('date.csv', index=False)