#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @时间    : 2018年3月2日 18:33
# @创建人  : Kchen
# @作用  : 最优分箱 - 最优降基
"""

import pandas as pd
import numpy as np
from sklearn.utils.multiclass import type_of_target

desired_width = 320
pd.set_option('display.width', desired_width)


class AutoSplit:
    """自动化分组:给定df，给出除了目标变量以外的全部离散化分组"""

    def __init__(self, method=1, max=5):
        self.method = method  # 默认的方法
        self.max = max  # 默认的最大分组
        self.acc = 0.01  # 初始分组精度
        self.target = 1  # 目标标志
        self.adjust = 0.000001  # 默认的调整系数

    def bin_for_continues(self, x, y):
        """
        连续变量的最优分组
        方法由method指定 :Gini, 2:Entropy, 3:person chisq, 4:Info value
        """

        # 检查y是否是二分类变量
        self._check_target_binary(y)

        # 等宽分组- 默认将数据切分为100份
        bin_map = self._equal_width_bin_map(x)

        # 把x映射到分组
        bin_res = self.apply_bin_for_continues(x, bin_map)

        # 合并x,y,以及对应的分组
        temp_df = pd.concat([x, y, bin_res], axis=1)

        # 每组内的0类，1类的数量，总数量
        t1 = pd.crosstab(index=temp_df[bin_res.name], columns=y)
        t2 = temp_df.groupby(bin_res.name).count().ix[:, 0]
        t2 = pd.DataFrame(t2)
        t2.columns = ['total']
        t = pd.concat([t1, t2], axis=1)
        temp_cont = pd.merge(t, bin_map,
                             left_index=True,
                             right_index=True,
                             how='left')

        temp_cont['pdv1'] = temp_cont.index

        # 合并组别，如果任意(0,1,total)==0，合并组别
        temp_cont = self._combine_bins(temp_cont)

        # 根据最大分区个数，重新计算分区
        temp_cont['bin'] = 1
        temp_cont['i'] = range(1, len(temp_cont) + 1)
        temp_cont['var'] = temp_cont.index
        nbins = 1
        while (nbins < self.max):
            temp_cont = self._cand_split(temp_cont)
            nbins += 1

        temp_cont = temp_cont.rename(columns={'var': 'oldbin'})
        temp_map = temp_cont.drop([0, self.target, 'pdv1', 'i'], axis=1)
        temp_map = temp_map.sort_values(by=['bin', 'oldbin'])
        # get new lower, upper, bin, total for sub
        data = pd.DataFrame()
        s = set()
        for i in temp_map['bin']:
            if i in s:
                pass
            else:
                sub_Map = temp_map[temp_map['bin'] == i]
                rowdata = self._get_new_bins(sub_Map, i)
                data = data.append(rowdata, ignore_index=True)
                s.add(i)

        # resort data
        data = data.sort_values(by='lower')
        data['newbin'] = range(1, self.max + 1)
        data = data.drop('bin', axis=1)
        data.index = data['newbin']
        data = data.rename(columns={'newbin': 'bin'})
        return data

    def bin_for_category(self, x, y):
        """
        分类变量的合并
        Reduce category for x by y & method
        method is represent by number,
            1:Gini, 2:Entropy, 3:person chisq, 4:Info value
        ----------------------------------------------
        Params:
        x: pandas Series, which need to reduce category
        y: pandas Series, 0-1 distribute dependent variable
        ---------------------------------------------
        Return
        temp_cont: pandas dataframe, reduct category map
        """
        self._check_target_binary(y)
        temp_cont, m = self._group_cal(x, y)
        nbins = 1
        # 如果离散变量本身的组数少于最大分组数量
        if m <= self.max:
            temp_cont = temp_cont.sort_index()
            temp_cont['bin'] = np.arange(1,m+1)
        else:
            while (nbins < self.max):
                temp_cont = self._cand_split(temp_cont)
                nbins += 1

        temp_cont = temp_cont.rename(columns={'var': x.name})
        temp_cont = temp_cont.drop([0, 1, 'i', 'pdv1'], axis=1)
        temp_cont = temp_cont.sort_values(by='bin')
        return temp_cont

    def _check_target_binary(self, y):
        """
        检查目标变量y是否是二分类变量
        :param y:
        :return:
        """
        y_type = type_of_target(y)
        if y_type not in ['binary']:
            raise ValueError('目标变量必须是二元的！')

    def _check_x_null_zero(self, x):
        """
        检查x是否为0
        """
        cond1 = np.isnan(x)
        cond2 = x == 0
        return cond1 or cond2

    def _equal_width_bin_map(self, x):
        """
        等宽分组
        :param x: pd.Series
        :return:
        """
        # 数据的最大值、最小值
        var_max = x.max()
        var_min = x.min()
        # 分组数量
        m_bins = int(1. / self.acc)
        min_max_size = (var_max - var_min) / m_bins
        ind = range(1, m_bins + 1)
        upper = pd.Series(index=ind, name='upper')
        lower = pd.Series(index=ind, name='lower')
        for i in ind:
            upper[i] = var_min + i * min_max_size
            lower[i] = var_min + (i - 1) * min_max_size

        # 调整分区的最小值，最大值
        upper[m_bins] = upper[m_bins] + self.adjust
        lower[1] = lower[1] - self.adjust
        bin_map = pd.concat([lower, upper], axis=1)
        bin_map.index.name = 'bin'
        return bin_map

    def apply_bin_for_continues(self, x, bin_map):
        """
        连续变量映射分组
        ------------------------------------------------
        Params
        x: pandas Series
        bin_map: pandas dataframe
        ------------------------------------------------
        Return
        bin_res: pandas Series, result of bining
        """
        bin_res = np.array([0] * x.shape[-1], dtype=int)

        for i in bin_map.index:
            upper = bin_map['upper'][i]
            lower = bin_map['lower'][i]
            x1 = x[np.where((x >= lower) & (x <= upper))[0]]
            mask = np.in1d(x, x1)
            bin_res[mask] = i

        bin_res = pd.Series(bin_res, index=x.index)
        bin_res.name = x.name + "_BIN"

        return bin_res

    def _cand_split(self, bin_ds):
        """
        Generate all candidate splits from current Bins
        and select the best new bins
        middle procession functions for bin_cont_var & reduce_cats
        ---------------------------------------------
        Params
        bin_ds: pandas dataframe, middle bining table
        method: int obj, metric to split x
            (1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
        --------------------------------------------
        Return
        newBins: pandas dataframe, split results
        """
        # sorted data by bin&pdv1
        bin_ds = bin_ds.sort_values(by=['bin', 'pdv1'])
        # get the maximum of bin
        Bmax = max(bin_ds['bin'])
        # screen data and cal nrows by diffrence bin
        # and save the results in dict
        temp_binC = dict()
        m = dict()
        for i in range(1, Bmax + 1):
            temp_binC[i] = bin_ds[bin_ds['bin'] == i]
            m[i] = len(temp_binC[i])
        """
        CC
        """
        # create null dataframe to save info
        temp_trysplit = dict()
        temp_main = dict()
        bin_i_value = []
        for i in range(1, Bmax + 1):
            if m[i] > 1:  # if nrows of bin > 1
                # split data by best i
                temp_trysplit[i] = self._best_split(temp_binC[i], i)
                temp_trysplit[i]['bin'] = np.where(temp_trysplit[i]['split'] == 1,
                                                   Bmax + 1,
                                                   temp_trysplit[i]['bin'])
                # delete bin == i
                temp_main[i] = bin_ds[bin_ds['bin'] != i]
                # vertical combine temp_main[i] & temp_trysplit[i]
                temp_main[i] = pd.concat([temp_main[i], temp_trysplit[i]], axis=0)
                # calculate metric of temp_main[i]
                value = self._g_value(temp_main[i])
                newdata = [i, value]
                bin_i_value.append(newdata)
        # find maxinum of value bintoSplit
        bin_i_value.sort(key=lambda x: x[1], reverse=True)
        # binNum = temp_all_Vals['BinToSplit']
        binNum = bin_i_value[0][0]
        newBins = temp_main[binNum].drop('split', axis=1)
        return newBins.sort_values(by=['bin', 'pdv1'])

    def _best_split(self, bin_ds, bin_no):
        """
        find the best split for one bin dataset
        middle procession functions for _cand_split
        --------------------------------------
        Params
        bin_ds: pandas dataframe, middle bining table
        method: int obj, metric to split x
            (1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
        bin_no: int obj, bin number of bin_ds
        --------------------------------------
        Return
        newbin_ds: pandas dataframe
        """
        bin_ds = bin_ds.sort_values(by=['bin', 'pdv1'])
        mb = len(bin_ds[bin_ds['bin'] == bin_no])

        bestValue = 0
        bestI = 1
        for i in range(1, mb):
            # split data by i
            # metric: Gini,Entropy,pearson chisq,Info value
            value = self._cal_cmerit(bin_ds, i)
            # if value>bestValue，then make value=bestValue，and bestI = i
            if bestValue < value:
                bestValue = value
                bestI = i
        # create new var split
        bin_ds['split'] = np.where(bin_ds['i'] <= bestI, 1, 0)
        bin_ds = bin_ds.drop('i', axis=1)
        newbin_ds = bin_ds.sort_values(by=['split', 'pdv1'])
        # rebuild var i
        newbin_ds_0 = newbin_ds[newbin_ds['split'] == 0].copy()
        newbin_ds_1 = newbin_ds[newbin_ds['split'] == 1].copy()
        newbin_ds_0['i'] = range(1, len(newbin_ds_0) + 1)
        newbin_ds_1['i'] = range(1, len(newbin_ds_1) + 1)
        newbin_ds = pd.concat([newbin_ds_0, newbin_ds_1], axis=0)
        return newbin_ds  # .sort_values(by=['split','pdv1'])

    def _g_value(self, bin_ds):
        """
        计算当前分段下的系数
        ----------------------------------------
        Params
        bin_ds: 数据框
        method: int 类型：1:Gini,         2:Entropy,
                         3:person chisq, 4:Info value
        -----------------------------------------
        Return
        M_value: float 或者 np.nan
        """
        R = bin_ds['bin'].max()
        N = bin_ds['total'].sum()

        N_mat = np.empty((R, 3))
        # calculate sum of 0,1
        N_s = [bin_ds[0].sum(), bin_ds[1].sum()]
        # calculate each bin's sum of 0,1,total
        # store values in R*3 ndarray
        for i in range(int(R)):
            subDS = bin_ds[bin_ds['bin'] == (i + 1)]
            N_mat[i][0] = subDS[0].sum()
            N_mat[i][1] = subDS[1].sum()
            N_mat[i][2] = subDS['total'].sum()

        # Gini
        if self.method == 1:
            G_list = [0] * R
            for i in range(int(R)):

                for j in range(2):
                    G_list[i] = G_list[i] + N_mat[i][j] * N_mat[i][j]
                G_list[i] = 1 - G_list[i] / (N_mat[i][2] * N_mat[i][2])
            G = 0
            for j in range(2):
                G = G + N_s[j] * N_s[j]

            G = 1 - G / (N * N)
            Gr = 0
            for i in range(int(R)):
                Gr = Gr + N_mat[i][2] * (G_list[i] / N)
            M_value = 1 - Gr / G
        # Entropy
        elif self.method == 2:
            for i in range(int(R)):
                for j in range(2):
                    if np.isnan(N_mat[i][j]) or N_mat[i][j] == 0:
                        M_value = 0

            E_list = [0] * R
            for i in range(int(R)):
                for j in range(2):
                    E_list[i] = E_list[i] - ((N_mat[i][j] / float(N_mat[i][2])) \
                                             * np.log(N_mat[i][j] / N_mat[i][2]))

                E_list[i] = E_list[i] / np.log(2)  # plus
            E = 0
            for j in range(2):
                a = (N_s[j] / N)
                E = E - a * (np.log(a))

            E = E / np.log(2)
            Er = 0
            for i in range(2):
                Er = Er + N_mat[i][2] * E_list[i] / N
            M_value = 1 - (Er / E)
            return M_value
        # Pearson X2
        elif self.method == 3:
            N = N_s[0] + N_s[1]
            X2 = 0
            M = np.empty((R, 2))
            for i in range(int(R)):
                for j in range(2):
                    M[i][j] = N_mat[i][2] * N_s[j] / N
                    X2 = X2 + (N_mat[i][j] - M[i][j]) * (N_mat[i][j] - M[i][j]) / (M[i][j])

            M_value = X2
        # Info value
        else:
            if any([self._check_x_null_zero(N_mat[i][0]),
                    self._check_x_null_zero(N_mat[i][1]),
                    self._check_x_null_zero(N_s[0]),
                    self._check_x_null_zero(N_s[1])]):
                M_value = np.NaN
            else:
                IV = 0
                for i in range(int(R)):
                    IV = IV + (N_mat[i][0] / N_s[0] - N_mat[i][1] / N_s[1]) \
                         * np.log((N_mat[i][0] * N_s[1]) / (N_mat[i][1] * N_s[0]))
                M_value = IV

        return M_value

    def _cal_cmerit(self, temp, ix):
        """
        Calculation of the merit function for the current table temp
        ---------------------------------------------
        Params
        temp: pandas dataframe, temp table in _best_split
        ix: single int obj,index of temp, from length of temp
        method: int obj, metric to split x(1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
        ---------------------------------------------
        Return
        M_value: float or np.nan
        """
        # split data by ix
        temp_L = temp[temp['i'] <= ix]
        temp_U = temp[temp['i'] > ix]
        # calculate sum of 0, 1, total for each splited data
        n_11 = float(sum(temp_L[0]))
        n_12 = float(sum(temp_L[1]))
        n_21 = float(sum(temp_U[0]))
        n_22 = float(sum(temp_U[1]))
        n_1s = float(sum(temp_L['total']))
        n_2s = float(sum(temp_U['total']))
        # calculate sum of 0, 1 for whole data
        n_s1 = float(sum(temp[0]))
        n_s2 = float(sum(temp[1]))
        N_mat = np.array([[n_11, n_12, n_1s],
                          [n_21, n_22, n_2s]])
        N_s = [n_s1, n_s2]
        # Gini
        if self.method == 1:
            N = n_1s + n_2s
            G1 = 1 - ((n_11 * n_11 + n_12 * n_12) / float(n_1s * n_1s))
            G2 = 1 - ((n_21 * n_21 + n_22 * n_22) / float(n_2s * n_2s))
            G = 1 - ((n_s1 * n_s1 + n_s2 * n_s2) / float(N * N))
            M_value = 1 - ((n_1s * G1 + n_2s * G2) / float(N * G))
        # Entropy
        elif self.method == 2:
            N = n_1s + n_2s
            E1 = -((n_11 / n_1s) * (np.log((n_11 / n_1s))) + \
                   (n_12 / n_1s) * (np.log((n_12 / n_1s)))) / (np.log(2))
            E2 = -((n_21 / n_2s) * (np.log((n_21 / n_2s))) + \
                   (n_22 / n_2s) * (np.log((n_22 / n_2s)))) / (np.log(2))
            E = -(((n_s1 / N) * (np.log((n_s1 / N))) + ((n_s2 / N) * \
                                                        np.log((n_s2 / N)))) / (np.log(2)))
            M_value = 1 - (n_1s * E1 + n_2s * E2) / (N * E)
        # Pearson chisq
        elif self.method == 3:
            N = n_1s + n_2s
            X2 = 0
            M = np.empty((2, 2))
            for i in range(2):
                for j in range(2):
                    M[i][j] = N_mat[i][2] * N_s[j] / N
                    X2 = X2 + ((N_mat[i][j] - M[i][j]) * (N_mat[i][j] - M[i][j])) / M[i][j]

            M_value = X2
        # Info Value
        else:
            try:
                IV = ((n_11 / n_s1) - (n_12 / n_s2)) * np.log((n_11 * n_s2) / (n_12 * n_s1)) \
                     + ((n_21 / n_s1) - (n_22 / n_s2)) * np.log((n_21 * n_s2) / (n_22 * n_s1))
                M_value = IV
            except ZeroDivisionError:
                M_value = np.nan
        return M_value

    def _combine_bins(self, temp_cont):
        """
        合并区间：如果0,1,total有一项为0，则合并分区
        ---------------------------------
        Params
        temp_cont: pandas dataframe
        target: target label
        --------------------------------
        Return
        temp_cont: pandas dataframe
        """
        drop_in = []
        len_index = len(temp_cont.index)
        for i in range(len_index):
            # 获取temp_cont中的一行数据
            rowdata = temp_cont.iloc[i, :]

            # 最后一行
            if i == len_index - 1:
                ix = i - 1
                while any(temp_cont.iloc[ix, :3] == 0):
                    ix = ix - 1
            else:
                ix = i + 1

            if any(rowdata[:3] == 0):  # 如果0,1,total有一项为0，则运行
                # 0、1、total 三列做累加操作
                target = self.target
                temp_cont.iloc[ix, target] = temp_cont.iloc[ix, target] + rowdata[target]
                temp_cont.iloc[ix, 0] = temp_cont.iloc[ix, 0] + rowdata[0]
                temp_cont.iloc[ix, 2] = temp_cont.iloc[ix, 2] + rowdata[2]
                drop_in.append(temp_cont.index[i])

                # 重新设定下限，上限
                if i == len_index - 1:
                    temp_cont.iloc[ix, 3] = rowdata['lower']
                    temp_cont.iloc[ix, 4] = rowdata['upper']
                elif i < temp_cont.index.max():
                    temp_cont.iloc[ix, 3] = rowdata['lower']
                else:
                    temp_cont.iloc[ix, 4] = rowdata['upper']
        temp_cont = temp_cont.drop(drop_in, axis=0)

        return temp_cont.sort_values(by='pdv1')

    def _get_new_bins(self, sub, i):
        """
        获取新的分段区间，也就是最终的分段
        -----------------------------------------
        Params
        sub: pandas dataframe, subdataframe of temp_map
        i: int, bin number of sub
        ----------------------------------------
        Return
        df: pandas dataframe, one row
        """
        l = len(sub)
        total = sub['total'].sum()
        first = sub.iloc[0, :]
        last = sub.iloc[l - 1, :]

        lower = first['lower']
        upper = last['upper']
        df = pd.DataFrame()
        df = df.append([i, lower, upper, total], ignore_index=True).T
        df.columns = ['bin', 'lower', 'upper', 'total']
        return df

    def _group_cal(self, x, y):
        """
        group calulate for x by y
        -------------------------------------
        Params
        x: pandas Series, which need to reduce category
        y: pandas Series, 0-1 distribute dependent variable
        ------------------------------------
        Return
        temp_cont: group calulate table
        m: nrows of temp_cont
        """

        temp_cont = pd.crosstab(index=x, columns=y, margins=False)
        temp_cont['total'] = temp_cont.sum(axis=1)
        temp_cont['pdv1'] = temp_cont[self.target] / temp_cont['total']

        temp_cont['i'] = range(1, temp_cont.shape[0] + 1)
        temp_cont['bin'] = 1
        m = temp_cont.shape[0]
        return temp_cont, m

    def apply_bin_for_category(self, x, bin_map):
        """
        单个离散变量分组
        convert x to newbin by bin_map
        ------------------------------
        Params
        x: pandas Series
        bin_map: pandas dataframe, mapTable contain new bins
        ------------------------------
        Return
        new_x: pandas Series, convert results
        """
        try:
            d = dict()
            for i in bin_map.index:
                value = bin_map.loc[i, 'bin']
                d[i] = value

            bin_res = x.map(d)
            bin_res.name = x.name + '_BIN'
        except:
            x_name = x.name
            x = pd.DataFrame(x)
            bin_res = pd.merge(x, bin_map, how='left', left_on=x_name, right_index=True)['bin']
            bin_res.name = x_name + '_BIN'
        return bin_res

    #
    #
    # def table_translate(red_map):
    #     """
    #     table tranlate for red_map
    #     ---------------------------
    #     Params
    #     red_map: pandas dataframe,reduce_cats results
    #     ---------------------------
    #     Return
    #     res: pandas series
    #     """
    #     l = red_map['bin'].unique()
    #     res = pd.Series(index=l)
    #     for i in l:
    #         value = red_map[red_map['bin'] == i].index
    #         value = list(value.map(lambda x: str(x) + ';'))
    #         value = "".join(value)
    #         res[i] = value
    #     return res


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_iris

    iris = load_iris()
    y = np.where(iris.target > 1, 1, 0)
    X = pd.DataFrame(iris.data)
    X.columns = [i.replace('(cm)', '').replace(' ', '_') for i in iris.feature_names]
    X = X["sepal_width_"]

    split = AutoSplit(method=4, max=5)

    # 连续变量分组示例
    bin_map = split.bin_for_continues(X, pd.Series(y))
    print(bin_map)
    df = split.apply_bin_for_continues(X, bin_map)
    print(df.head())

    # 离散变量分组示例-需要加入判断逻辑
    X = pd.cut(X, 2)
    bin_map2 = split.bin_for_category(X, pd.Series(y))
    print(bin_map2.sort_index())
    df2 = split.apply_bin_for_category(X, bin_map2)
    print(df2.head())
