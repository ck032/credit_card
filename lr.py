#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @时间    : 2018年3月2日 16:36
# @创建人  : Kchen
# @作用  :
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas import Series, DataFrame
from datetime import datetime

desired_width = 320
pd.set_option('display.width', desired_width)


class Logistic:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.results = None
        self.model = None

    def model_fit(self, constant=True, stepwise=None, sls=0.05, alpha=None, verbose=True):
        """
        模型拟合
        """
        X = self.X
        y = self.y

        # 变量筛选-前向选择 VS 后向选择
        if stepwise == "FS" and X.shape[1] > 1:
            varlist = self._forward_selected_logit(X, y)
            X = X.ix[:, varlist]
        elif stepwise == "BS" and X.shape[1] > 1:
            varlist = self._backward_selected_logit(X, y, sls=sls)
            X = X.ix[:, varlist]

        # 截距项-只包含截距项拟合 VS 只带有截距项的拟合
        if constant:
            X = sm.add_constant(X)

        model = sm.Logit(y, X, missing='drop')

        # 拟合带有正则项的逻辑回归方程，alpha=0.0001
        if alpha != None:
            results = model.fit_regularized(alpha=alpha)
        else:
            results = model.fit()

        # 展示模型拟合结果
        if verbose:
            result_summary = results.summary()
            print(result_summary)
            for idx, table in enumerate(result_summary.tables):
                print('保存第{}个结果'.format(idx))
                with open("%s_result.csv" % idx, "w") as fcsv:
                    fcsv.write(table.as_csv())
        self.results = results
        self.model = model
        return model, results

    def model_desc(self):
        """
        模型拟合信息
        """

        rlt = {
            # 1.入参
            "模型": "二元logistic模型",
            "使用的观测个数": self.results.nobs,
            "含缺失值观测个数": self.X.shape[0] - self.results.nobs,
            "总观测个数": self.X.shape[0],
            "自变量": list(self.X.columns),

            # 2.似然比
            "似然比": self.results.llr,
            "自由度": self.results.df_model,
            "似然比p值": self.results.llr_pvalue,

            # 3.模型残差
            "残差": self.results.resid_generalized,
            "标准化残差": self.results.resid_pearson,
            "resid_response": self.results.resid_response,

            # 4.AIC,BIC,似然函数值
            'aic': self.results.aic,
            'bic': self.results.bic,
            '-2*logL': -2 * self.results.llf,
            "伪R方": self.results.prsquared,

            # 5.其他信息
            "方法": "最大似然估计",
            "日期时间": datetime.now()
        }
        return rlt

    def confusion_matrix(self):
        """
        混淆矩阵
        """
        cm = DataFrame(self.results.pred_table())
        cm.index.name = '实际结果'
        cm.columns.name = '预测结果'
        return cm

    def cov_matrix(self, normalized=False):
        """
        cov_matrix
        """
        if normalized:
            rlt = self.results.normalized_cov_params
        else:
            rlt = self.results.cov_params()
        return rlt

    def model_predict(self, X=None):
        """
        模型预测
        """
        pred = self.results.predict(X)
        pred = Series(pred)
        return pred

    def _forward_selected_logit(self, X, y):
        """
         evaluated by adjusted R-squared
        """
        import statsmodels.formula.api as smf
        data = pd.concat([X, y], axis=1)
        response = y.columns[0]
        remaining = set(X.columns)
        selected = []
        current_score, best_new_score = 0.0, 0.0
        while remaining and current_score == best_new_score:
            scores_with_candidates = []
            for candidate in remaining:
                formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
                print(formula)
                mod = smf.logit(formula, data).fit()
                score = mod.prsquared
                scores_with_candidates.append((score, candidate))
            scores_with_candidates.sort(reverse=False)
            best_new_score, best_candidate = scores_with_candidates.pop()
            if current_score < best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
        return selected

    def _backward_selected_logit(self, X, y, sls=0.05):
        """
        backward selection.
        sls: 根据p值的大小筛选
        """
        import statsmodels.formula.api as smf  # 导入相应模块
        data = pd.concat([X, y], axis=1)  # 合并数据
        # 提取X，y变量名
        var_list = X.columns
        response = y.columns[0]
        # 首先对所有变量进行模型拟合
        while True:
            formula = "{} ~ {} + 1".format(response, ' + '.join(var_list))
            mod = smf.logit(formula, data).fit()
            p_list = mod.pvalues.sort_values()
            if p_list[-1] > sls:
                # 提取p_list中最后一个index
                var = p_list.index[-1]
                # var_list中删除
                var_list = var_list.drop(var)
            else:
                break
        return var_list

if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    X = pd.read_csv("X.csv",index_col=0)
    y = pd.read_csv("y.csv",index_col=0)
    lr = Logistic(X,y)
    # lr.model_fit(stepwise='BS')
    lr.model_fit(alpha=0.00001)