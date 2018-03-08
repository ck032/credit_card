# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:49:17 2017

@author: Hank Kuang
@title: 评分卡生成
"""

import numpy as np
import statsmodels.api as sm
# import re
import pandas as pd


def creditCards(paramsEst,
                woe_maps,
                bin_maps,
                red_maps,
                basepoints=600,
                odds=60,
                PDO=20):
    """
    output credit card for each var in model
    --------------------------------------------
    ParamsEst: pandas Series, params estimate results in logistic model,
               index is param names,value is  estimate results
    bin_maps: dict, key is varname of paramEst , value is pandas dataframe contains map table
    red_maps: dict, key is varname of paramEst, value is pandas dataframe contains redCats table
    woe_maps: dict, key is varname of paramEst, value is also a dict contain binNumber--woe
    basepoints: expect base points
    odds: reciprocal of Odds
    PDO: double coefficients
    -------------------------------------------
    Return
    creditCard: pandas dataframe
    """
    # 计算A&B
    alpha, beta = _score_cal(basepoints, odds, PDO)
    # 计算基础分
    points_0 = round(alpha - beta * paramsEst['const'])
    # 根据各段woe，计算相应得分
    points = pd.DataFrame()
    for k in woe_maps.keys():
        d = pd.DataFrame(woe_maps[k], index=[k]).T

        d['points'] = round(-beta * d.ix[:, k] * paramsEst[k])
        if k in bin_maps.keys():
            bin_map = bin_maps[k]
            bin_map = bin_map.drop(['total', 'newbin'], axis=1)
            bin_map['range'] = bin_map.apply(lambda x: str(x[0]) + '--' + str(x[1]), axis=1)
            bin_map = bin_map.drop(['lower', 'upper'], axis=1)
            d = pd.merge(d, bin_map, left_index=True, right_index=True)

        elif k in red_maps.keys():
            red_map = red_maps[k]
            s = tableTranslate(red_map)
            s = pd.DataFrame(s.T, columns=['range'])
            d = pd.merge(d, s, left_index=True, right_index=True)

        else:
            d['range'] = d.index

        n = len(d)
        ind_0 = []
        i = 0
        while i < n:
            ind_0.append(k)
            i += 1

        d.index = [ind_0, list(d.index)]
        d = d.drop(k, axis=1)
        points = pd.concat([points, d], axis=0)

    # 输出评分卡
    points_0 = pd.DataFrame([[points_0, '-']],
                            index=[['basePoints'], ['-']],
                            columns=['points', 'range'])
    credit_card = pd.concat([points_0, points], axis=0)
    credit_card.index.names = ["varname", "binCode"]
    return credit_card


def tableTranslate(red_map):
    """
    table tranlate for red_map
    ---------------------------
    Params
    red_map: pandas dataframe,reduceCats results
    ---------------------------
    Return
    res: pandas series
    """
    l = red_map['bin'].unique()
    res = pd.Series(index=l)
    for i in l:
        value = red_map[red_map['bin'] == i].index
        value = list(value.map(lambda x: str(x) + ';'))
        value = "".join(value)
        res[i] = value
    return res


def _score_cal(basepoints, odds, PDO):
    """
    cal alpha&beta for score formula,
    score = alpha + beta * log(odds)
    ---------------------------------------
    Params
    basepoints: expect base points
    odds: cal by logit model
    PDO: points of double odds
    ---------------------------------------
    Return
    alpha, beta
    """
    beta = PDO / np.log(2)
    alpha = basepoints - beta * np.log(odds)
    return alpha, beta
