# from sympy import Symbol, solve, S
# from sympy.solvers import solveset
#
# x = Symbol('x', real=True)
# f = x ** 2 - 5 * x + 1
# # a = solve(f, x)
# result = solveset(f, x, domain=S.Reals)
#
# for i in result:
#     print(float(i))
# a = ['123', '111111', 1, 1, 838, 453]


import sys
from sympy import Symbol, solve

def solver(line):
    user_id, bid, type, i, ori_amount, total_earning = line.replace(' ', '').split(",")
    c = float(total_earning) / float(ori_amount)
    i = float(i)
    p = Symbol('p', real=True)
    tol = ((i * p - 1) * (1 + p) ** i + 1) / ((1 + p) ** i - 1) - c
    # result = solveset(tol, p, domain=S.Reals)
    result = solve(tol, p)
    for i in result:
        try:
            i = float(i)
            if i >= 0.01 and i <= 0.8:
                x = i
            else:
                x = 'can not solve'
        except:
            x = 'can not solve'
    return str(x)


try:
    for line in sys.stdin:
        line = line.strip()
        print(line + "  " + solver(line))
except Exception as e:
    print('[PYTHON_MONTH_RATE]###%s:%s' % (line, e))

#
#
#
#
#
#
#
#
# import sys
# import numpy as np
# import pandas as pd
# from sympy import Symbol, S
# from sympy.solvers import solveset
#
#
# def solver(line):
#     user_id, bid, type, i, ori_amount, total_earning = line.strip().split("\t")
#     c = float(total_earning) / float(ori_amount)
#     i = float(i)
#     p = Symbol('p', real=True)
#     tol = ((i * p - 1) * (1 + p) ** i + 1) / ((1 + p) ** i - 1) - c
#     result = solve(tol, p)
#     for i in result:
#         try:
#             i = float(i)
#             if i >= 0.01 and i <= 0.8:
#                 x = i
#             else:
#                 x = 'can not solve'
#         except:
#             x = 'can not solve'
#     return str(x)
#
#
# try:
#     for line in sys.stdin:
#         line = line.strip()
#         print(line + "  " + solver(line))
# except Exception as e:
#     print('[PYTHON_MONTH_RATE]###%s:%s' % (line, e))
#
#
# def solver2(line):
#     user_id, bid, type, i, ori_amount, total_earning = line.strip().split("\t")
#     c = float(total_earning) / float(ori_amount)
#     i = float(i)
#     p = Symbol('p', real=True)
#     tol = abs(((i * p - 1) * (1 + p) ** i + 1) / ((1 + p) ** i - 1) - c)
#     result = solveset(tol, p, domain=S.Reals)
#     for i in result:
#         try:
#             i = float(i)
#             if i >= 0.01 and i <= 0.8:
#                 x = i
#             else:
#                 x = 'can not solve'
#         except:
#             x = 'can not solve'
#     return str(x)
#
#
# try:
#     for line in sys.stdin:
#         line = line.strip()
#         print(line + "	" + solver2(line))
# except Exception as e:
#     print('[PYTHON_MONTH_RATE]###%s:%s' % (line, e))
#
#
# def solver(line):
#     user_id, bid, type, i, ori_amount, total_earning = line.strip().split("\t")
#     c = float(total_earning) / float(ori_amount)
#     i = float(i)
#     m = 100000000000000
#     x = 0
#     p_list = np.arange(0.01, 0.8, 0.001)
#     for p in p_list:
#         tol = abs(((i * p - 1) * (1 + p) ** i + 1) / ((1 + p) ** i - 1) - c)
#         if tol <= m:
#             m = tol
#             x = p
#     return str(x)
#
#
# try:
#     for line in sys.stdin:
#         line = line.strip()
#         print(line + "	" + solver(line))
# except Exception as e:
#     print('[PYTHON_MONTH_RATE]###%s:%s' % (line, e))
