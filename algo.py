# 导入函数库
from jqdata import *
from jqfactor import *
import datetime as dt
import numpy as np
import pandas as pd
import time


# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    log.set_level('order', 'error')

    # 参数设置
    # 第几组
    g.group = 1
    # 因子名 
    g.factor = 'pb_ratio'
    # 因子是否取倒数
    g.if_backward = 1
    
    
    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
    # 开盘前运行
    run_monthly(before_market_open, monthday = 1, time='before_open', reference_security='000300.XSHG')
    # 开盘时运行
    run_monthly(market_open, monthday = 1, time='open', reference_security='000300.XSHG')


## 开盘前运行函数
def before_market_open(context):
    #设置滑点、手续费
    set_slip_fee(context)
    
    #取沪深300作为股票池
    all_stocks = get_index_stocks('000300.XSHG', date = context.current_dt)
    feasible_stocks = set_feasible_stocks(context, all_stocks)
    
    # 通过函数找到对应的因子值
    # 为了保险，date取前一天的值
    # 倒数按照需求设置
    factor = get_factor(g.factor,context.current_dt - dt.timedelta(days=1),g.if_backward)

    #去极值
    factor = winsorize(factor, scale = 3, axis = 0)
    #中性化
    factor = neutralize(factor, how = ['sw_l1', 'market_cap'], date = context.current_dt, axis = 0, fillna = 'sw_l1')
    #标准化
    factor = standardlize(factor, axis = 0)

    #排序
    # 如果是python 3 取下面的值
    # factor = factor.sort_values('BP', ascending = True)
    factor = factor.sort(g.factor, ascending = True)

    n = int(len(factor)/10)
    #分组取样
    if g.group == 10:
        g.tobuy_list = factor.index[(g.group - 1) * n :]
    else:
        g.tobuy_list = factor.index[(g.group - 1) * n : g.group * n]


#1
#设置可行股票池，剔除(金融类、)st、停牌股票，输入日期
def set_feasible_stocks(context,s):
    #s = get_index_stocks('000905.XSHG', date=context.current_dt)
    #print '输入股票个数为：%s'%len(s)
    all_stocks = s
    #得到是否停牌信息的dataframe，停牌得1，未停牌得0
    suspended_info_df = get_price(list(all_stocks), end_date = context.current_dt, count = 1, frequency = 'daily', fields = 'paused')['paused'].T
    #过滤未停牌股票 返回dataframe
    suspended_index = suspended_info_df.iloc[:,0] == 1
    #得到当日停牌股票的代码list:
    suspended_stocks = suspended_info_df[suspended_index].index.tolist()

    #剔除停牌股票
    for stock in suspended_stocks:
        if stock in all_stocks:
            all_stocks.remove(stock)

    return all_stocks   



# 根据不同的时间段设置滑点与手续费
def set_slip_fee(context):
    # 将滑点设置为0
    set_slippage(FixedSlippage(0)) 
    # 根据不同的时间段设置手续费
    dt=context.current_dt

    if dt>datetime.datetime(2013,1, 1):
        set_commission(PerTrade(buy_cost=0.0003, 
                                sell_cost=0.0013, 
                                min_cost=5)) 

    elif dt>datetime.datetime(2011,1, 1):
        set_commission(PerTrade(buy_cost=0.001, 
                                sell_cost=0.002, 
                                min_cost=5))

    elif dt>datetime.datetime(2009,1, 1):
        set_commission(PerTrade(buy_cost=0.002, 
                                sell_cost=0.003, 
                                min_cost=5))

    else:
        set_commission(PerTrade(buy_cost=0.003, 
                                sell_cost=0.004, 
                                min_cost=5))


## 开盘时运行函数
def market_open(context):
    #调仓，先卖出股票
    for stock in context.portfolio.long_positions:
        if stock not in g.tobuy_list:
            order_target_value(stock, 0)

    #再买入新股票
    total_value = context.portfolio.total_value # 获取总资产
    for i in range(len(g.tobuy_list)):
        value = total_value / len(g.tobuy_list) # 确定每个标的的权重

        order_target_value(g.tobuy_list[i], value) # 调整标的至目标权重

    #查看本期持仓股数
    print(len(context.portfolio.long_positions))



## 收盘后运行函数
def after_market_close(context):
    pass


def get_factor(factor_name,date,if_backward):
    #获取五张财务基础所有指标名称
    val = get_fundamentals(query(valuation).limit(1)).columns.tolist()
    bal = get_fundamentals(query(balance).limit(1)).columns.tolist()
    cf = get_fundamentals(query(cash_flow).limit(1)).columns.tolist()
    inc = get_fundamentals(query(income).limit(1)).columns.tolist()
    ind = get_fundamentals(query(indicator).limit(1)).columns.tolist()

    stock = get_index_stocks('000300.XSHG', date)

    if factor_name in val:
        q = query(valuation).filter(valuation.code.in_(stock))
        df = get_fundamentals(q, date)
        
    elif factor_name in bal:
        q = query(balance).filter(balance.code.in_(stock))
        df = get_fundamentals(q, date)
        
    elif factor_name in cf:
        q = query(cash_flow).filter(cash_flow.code.in_(stock))
        df = get_fundamentals(q, date)


    elif factor_name in inc:
        q = query(income).filter(income.code.in_(stock))
        df = get_fundamentals(q, date)
    
    elif factor_name in ind:
        q = query(indicator).filter(indicator.code.in_(stock))
        df = get_fundamentals(q, date)
        

    ret_pd = pd.DataFrame()

    if if_backward:
        ret_pd[factor_name] = np.array(1/df[factor_name])
        ret_pd['code'] = np.array(df['code'])
        ret_pd = ret_pd.set_index('code')
    else:
        ret_pd[factor_name] = np.array(df[factor_name])
        ret_pd['code'] = np.array(df['code'])
        ret_pd = ret_pd.set_index('code')

    return ret_pd