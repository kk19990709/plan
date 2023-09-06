import pandas as pd
import datetime

def percent(num):
    return num / 100

def repercent(num):
    return num * 100

def date2str(date):
    if type(date) == str: return date
    return datetime.datetime.strftime(date, '%Y-%m-%d')

def date_span(start_date, end_date):
    time_range = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    assert time_range.days >= 0
    return time_range.days + 1

def date_diff(start_date, end_date):
    time_range = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    assert time_range.days >= 0
    return time_range.days

def daily_interest(interest):
    return percent(interest) / 360

def annual_interest(interest):
    return repercent(interest * 360)