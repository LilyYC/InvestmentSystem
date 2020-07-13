# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:30:16 2020

Grab historical stock price from Yahoo! for a list of stocks
@author: lily
"""

import re
import requests
import pandas as pd
from datetime import datetime, date
import time

def yahoo_date(date1):
    temp = date(1970, 1, 1)
    delta = date1 - temp
    return delta.days * 86400 + delta.seconds

class Stock:
    def __init__(self, number):
        #self.name = name
        self.number = number
        self.exchange = self.number[-2:]
        self.curr = date.today()
        self.start = date(2015,1,1)

    def get_historical_price(self):
        from_date = yahoo_date(self.start)
        to_date = yahoo_date(self.curr)
        his_url = f'https://ca.finance.yahoo.com/quote/{self.number}/history?period1={from_date}&period2={to_date}&interval=1d&filter=history&frequency=1d'
        source = requests.get(his_url)
        html_str = source.content.decode()
        datastr = re.findall('"HistoricalPriceStore":{"prices"\:(.*?),"isPending"', html_str, re.S)[0]
        datalist = datastr.split('{')
        data_value = []
        #data_date = []
        for i in range(1, len(datalist)-1):
            list_price = re.findall('"adjclose":(.*?)},',datalist[i], re.S)
            if list_price !=  [] and list_price != 'null':
                p = float(list_price[0])
                t = re.findall('"date":(.*?),"', datalist[i], re.S)[0]
                d = datetime.fromtimestamp(int(t)).date()
                data_value.append([d,p])
                #data_date.append(d)
        df = pd.DataFrame(data_value, columns=['Date','Price'])
        #df = pd.DataFrame(data_value, index = data_date, columns=[self.number])
        return df
    
    def get_curr(self):
        return self.get_historical_price()[self.number][0]
    
    def get_return(self):
        pass #return np.ln(self.get_curr() - )
        
    def get_vol(self):
        his = self.get_historical_price()
        mu = his.mean()
        # vol = his-mu
