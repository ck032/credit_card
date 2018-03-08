#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @时间    : 2018年3月2日 16:39
# @创建人  : Kchen
# @作用  : WOE转换和信息值(info value)计算
"""

import pandas as pd
import numpy as np
import math
from sklearn.utils.multiclass import type_of_target
from collections import namedtuple


class WOE:

    def __init__(self):
        self.woe_dict = None
        self.iv_dict = None
        self._WOE_MIN = -10
        self._WOE_MAX = 10

    def __repr__(self):
        return "离散变量转化为WOE值，提供了woe_trans方法"

    def woe_trans(self, y, df):
        """
        全部变量的WOE转换
        """
        self.woe_dict = {}
        self.iv_dict = {}
        result = namedtuple('Result', ['df', 'woe_dict', 'iv_dict'])
        for var in df.columns:
            x = df[var]
            x_woe_trans, woe_map, info_value = self._single_woe_trans(x, y)
            df = pd.concat([df, x_woe_trans], axis=1)
            self.woe_dict[var] = woe_map
            self.iv_dict[var] = info_value
        self.iv_dict = sorted(self.iv_dict.items(), key=lambda p: p[1], reverse=True)
        return result(df, self.woe_dict, self.iv_dict)

    def _check_target_binary(self, y):
        """
        检查目标变量是否是二元变量
        """
        y_type = type_of_target(y)
        if y_type not in ['binary']:
            raise ValueError('目标变量必须是二元的！')

    def _count_binary(self, y, event=1):
        """
        目标值，非目标值的计数
        默认目标是y = 1
        """
        event_count = (y == event).sum()
        non_event_count = y.shape[-1] - event_count
        return event_count, non_event_count

    def _woe_single_x(self, x, y, event=1):
        """
        单个变量的WOE，IV值计算
        """

        # 检查目标变量是否是二元变量
        self._check_target_binary(y)

        # 目标计数
        event_total, non_event_total = self._count_binary(y, event=event)

        # 离散变量的唯一值
        x_labels = x.unique()

        # 每个变量的WOE值
        woe_dict = {}

        # 每个变量的IV值
        iv = 0
        for x1 in x_labels:
            y1 = y[np.where(x == x1)[0]]
            event_count, non_event_count = self._count_binary(y1, event=event)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            if rate_event == 0:
                woe = self._WOE_MIN
            elif rate_non_event == 0:
                woe = self._WOE_MAX
            else:
                woe = math.log(rate_event / rate_non_event)
            woe_dict[x1] = woe
            iv += (rate_event - rate_non_event) * woe
        return woe_dict, iv

    def _single_woe_trans(self, x, y):
        """
        单个变量的WOE转化
        """
        woe_map, info_value = self._woe_single_x(x, y)
        x_woe_trans = x.map(lambda x: woe_map[x])
        x_woe_trans.name = x.name + "_WOE"
        return x_woe_trans, woe_map, info_value


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_iris

    iris = load_iris()
    y = np.where(iris.target > 1, 1, 0)
    X = pd.DataFrame(iris.data).apply(lambda x: pd.cut(x, 5))
    X.columns = [i.replace('(cm)', '').replace(' ', '_') for i in iris.feature_names]
    WOE = WOE()
    result = WOE.woe_trans(y, X)

    df = result.df[['sepal_length__WOE','sepal_width__WOE','petal_length__WOE','petal_width__WOE']]
    df.to_csv("X.csv")
    pd.DataFrame(y).to_csv("y.csv")
    print(result.df.head())
    print(result.iv_dict)
