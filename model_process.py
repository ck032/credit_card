#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @时间    : 2018年3月6日 15:12
# @创建人  : Kchen
# @作用  : 完整的LR建模过程
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

desired_width = 320
pd.set_option('display.width', desired_width)


class ModelingProcess:
    def __init__(self, logit_config, dev, val, variable_list, target):
        self.var_group_config = logit_config  # 逻辑回归配置，包括变量类型，变量分箱函数
        self.dev_sample = dev  # 开发样本
        self.val_sample = val  # 测试样本
        self.var_list = variable_list  # 变量列表
        self.var_woe_list = []  # 变量woe名列表
        self.tgt = target  # 目标变量名
        self.var_grp_df_dict = {}  # 各变量分箱结果字典
        self.attr_score_dict = {}  # 变量分箱打分字典
        self.var_dict = {}  # 各变量信息表，包括原变量名，分箱函数，分箱结果
        self.var_iv = []
        self.std_offset = 60
        self.std_odds = 3.36
        self.pdo = 12
        self.offset = 0
        self.factor = 0
        self.log_model = object
        self.log_model_res = object
        self.dev_err_rate = 0
        self.val_err_rate = 0
        self.dev_gains = None
        self.val_gains = None

        pass
