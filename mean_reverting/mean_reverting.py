import os.path
import pandas as pd
import numpy as np
import datetime
import json
import statsmodels.api as sm
import re
import shutil
import warnings
import jqdatasdk
from jqdatasdk import *
import pymysql
import datetime
import math
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
#import multiprocessing


def get_path(path):
    path_to_get = path
    if not os.path.exists(path_to_get):
        os.makedirs(path_to_get)
        return path_to_get
    else:
        return path_to_get

def get_JQ_code(x):
    x_ = x
    if (x_[0] == '0') or (x_[0] == '3'):
        add_ = '.XSHE'
        return x_ + add_
    if x_[0] == '6':
        add_ = '.XSHG'
        return x_ + add_



class mean_reverting():
    def __init__(self):
        print(datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d"))
        jqdatasdk.auth('', '')
        with open('parameters.json', 'r') as f:
            self.parameters = json.load(f)

    def mean_reverting(self):
        jqdatasdk.auth('', '')
        print(datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d") + ' start mean reverting training')
        # assumption: 允许融券
        # target：判断买卖点
        root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        path_data = get_path(os.path.join(root,"data"," "))

        start_time_str = self.parameters['backtest_time_period'][0]
        end_time_str = self.parameters['backtest_time_period'][1]
        csi300 = jqdatasdk.get_price('000300.XSHG',start_time_str,end_time_str)
        if self.parameters['stock_name']['type_mode'] == 0:  # 五粮液和贵州茅台，白酒对，
            pairs_a = self.parameters['stock_name']['type_0'][0]  # 白酒行业 选择五粮液和贵州茅台作为套利对 贵州茅台
            pairs_b = self.parameters['stock_name']['type_0'][1]  # 五粮液
            dt = self.parameters['data_type'][0] # choose data type
            mt_price = jqdatasdk.get_price(get_JQ_code(pairs_a),start_time_str,end_time_str)[dt]
            wl_price = jqdatasdk.get_price(get_JQ_code(pairs_b),start_time_str,end_time_str)[dt]
            data_rw_wine = pd.concat([mt_price,wl_price],axis=1)
            data_rw_wine.columns = [pairs_a,pairs_b]
            data_rw_wine['csi300'] = csi300['high']
            data_rw_wine.to_csv(path_data+'raw_data_wine.csv',encoding='utf-8')
            # 首先考虑两个股票的股价的相关性
            # 绘制折线图
            data_plot = data_rw_wine
            data_plot /= data_plot.iloc[0]
            get_fig(data_plot,path_data,"wine_pnl","date","pnl","wine_pnl")
            # 判断两股票的相关性, 采取rolling方法
            rol_win = self.parameters['window']
            df = pd.concat([mt_price,wl_price],axis=1)
            df.columns=[pairs_a,pairs_b]
            # pandas 计算co-relation coefficiency
            df_corr = df.rolling(window=rol_win).corr()
            # 用IndexSlice 取multi index中的其中某一个index
            idx = pd.IndexSlice
            df_corr_2 = df_corr.loc[idx[:,'000858'],]['600519'].dropna()
            df_corr_2.columns=['co_relations_wine']
            get_fig(pd.DataFrame(df_corr_2).droplevel(1),path_data,"wine_corelations","date","co_relations","wine_corelations")
            # scipy 计算co-relation coeeficiency
            r, p = stats.pearsonr(df[pairs_a],df[pairs_b])
            sts_r = pd.DataFrame()
            sts_p = pd.DataFrame()
            # range 函数是从0开始
            for i in range(len(df)-rol_win+1):
                df_i = df.iloc[i:i+rol_win]
                r_i = stats.pearsonr(df_i[pairs_a],df_i[pairs_b])[0]
                p_i = stats.pearsonr(df_i[pairs_a],df_i[pairs_b])[1]
                df_r_i = pd.DataFrame([r_i],index=[df_i.index[-1]])
                df_p_i = pd.DataFrame([p_i],index=[df_i.index[-1]])
                sts_r = pd.concat([sts_r,df_r_i],axis=0)
                sts_p = pd.concat([sts_p,df_p_i],axis=0)
            sts_r.columns=['r']
            sts_p.columns=['p']
            sts_comb = pd.concat([sts_r,sts_p],axis=1)

            sts_comb.to_csv(path_data+'sts_comb_corr_p.csv',encoding='utf-8')
            # 绘图
            fig = plt.figure(figsize=(18,16))
            ax_0 = fig.add_subplot(211)
            ax_0.plot(sts_r)
            plt.grid(True)
            plt.legend(sts_r.columns,loc='best')
            # plt.xlabel("date",labelsize=10)
            # sts_r.plot(ax=ax[0])
            ax_0.set(xlabel="date",ylabel="co-relation coefficient")

            ax_1 = fig.add_subplot(212)
            ax_1.scatter(sts_p.index,sts_p,s=75,alpha=0.5)
            # sts_p.scatter(ax=ax[1])
            ax_1.set(xlabel="date",ylabel="p_value")
            ax_0.get_shared_x_axes().join(ax_0, ax_1)
            plt.savefig(path_data+'corr_comb_p.pdf')
            # plt.show()




        print('...')
            # 按照日内最高价来测试均值回归