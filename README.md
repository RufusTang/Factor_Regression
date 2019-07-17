# Factor_Regression
Test Factor Regression effect

# ARMA_Modeling.ipynb
通过检测ARMA的建模，可以检测出时间序列是否符合ARMA模型
通过建模，可以检测出序列究竟是不是随机序列，如果是随机序列，则P，Q都是0阶
进一步如果时序序列推算不出来，则傅里叶变换也是无效的
目前按照沪深300的序列，均是0阶，是无法通过时间序列模型套利

# Factor_Loading.ipyn
通过代码实现
1. 通过研究模块调用策略模块，设置不同的参数回测
2. 生成pandas数组，按照每月的数据计算因子载荷
3. 可以最后计算出因子载荷的平均值，标准差
下一步可以通过几种算法计算：计算时序平均值、通过ARMA模型预测

# algo.py
在策略模块实现的算法
有一个问题，通过策略模块调用会自动转换到python 2，如果使用python 3 编码，编译器是python 2会出错

# result_factor_pd_by_month_no_reverse.csv
月度因子溢价原始数值，该部分因子没有添加倒数

# result_factor_pd_by_month_reverse.csv
月度因子溢价的原始数值，该部分因子使用了倒数，主要为了避免负值的影响
