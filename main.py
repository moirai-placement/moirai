import gurobipy
from gurobipy import *
import datetime
import xlrd
import re

# read from excel data
excels = []


def print_xls(path):
    data = xlrd.open_workbook(path)  # open excel
    table = data.sheets()[0]  # open excel sheet
    nrows = table.nrows  # capture the rows with numbers 
    for i in range(nrows):
        ss = table.row_values(i)  # capture the value of a row, each column data is an item
        excels.append(ss)


print_xls('path/to/xls_file')

# parameter1
O_G = []  # task node
O_GG = []  # communication node
dirSucc_OG = []  # successor task in DAG(OG)
dirSucc_add_OG = []  # successor task considering communication task in DAG (OG+OGG)
dirSucc_OGG = [] 
K = 3  # number of devices
ptask = [[] for k in range(K)]  # processing time of task on K devices
pcomm = [[[] for k1 in range(K)] for k2 in range(K)]  # communication time
mtask = []  # footprint
Mem = []  # device memory
Ms = Ml = Mr = 100000  # large number
for row_len in range(len(excels)):
    if len(excels) - 1 > row_len > 0:  # virtual nodes
        # O_G
        O_G.append(int(excels[row_len][0]))
        # O_GG
        if type(excels[row_len][1]) == str:
            if excels[row_len][1] != '':
                O_GG.extend(excels[row_len][1].split(','))
                O_GG = list(map(int, O_GG))
        else:
            O_GG.append(int(excels[row_len][1]))
        # dirSucc_OGG
        if type(excels[row_len][2]) == str and row_len > 1:
            if excels[row_len][2] != '':
                dirSucc_OGG.extend(excels[row_len][2].split(','))
                dirSucc_OGG = list(map(int, dirSucc_OGG))
        elif row_len > 1:
            dirSucc_OGG.append(int(excels[row_len][2]))
        # dirSucc_OG
        if type(excels[row_len][2]) == str:
            if excels[row_len][2] != '':
                dirSucc_OG.append(list(map(int, excels[row_len][2].split(','))))
        else:
            dirSucc_OG.append([int(excels[row_len][2])])
        # dirSucc_add_OG
        if type(excels[row_len][3]) == str:
            if excels[row_len][3] != '':
                dirSucc_add_OG.append(list(map(int, excels[row_len][3].split(','))))
        else:
            dirSucc_add_OG.append([int(excels[row_len][3])])
        # processing time
        for k in range(K):
            ptask[k].append(float(excels[row_len][4 + k]))
        # pc
        count_k = 0
        for k1 in range(K):
            for k2 in range(K):
                if type(excels[row_len][1]) == str:
                    if excels[row_len][1] != '':
                        pcomm[k1][k2].extend(excels[row_len][4 + K + count_k].split(','))
                        pcomm[k1][k2] = list(map(float, pcomm[k1][k2]))
                else:
                    pcomm[k1][k2].append(float(excels[row_len][4 + K + count_k]))
                count_k += 1
        # m
        mtask.append(int(excels[row_len][4 + K + K * K]))

    elif row_len == len(excels) - 1:
        O_G.append(int(excels[row_len][0]))
        dirSucc_OG.append([])
        dirSucc_add_OG.append([])
        mtask.append(int(excels[row_len][4 + K + K * K]))
        for k in range(K):
            ptask[k].append(int(excels[row_len][4 + k]))
for k in range(K):
    Mem.append(int(excels[1][5 + K + K * K + k]))

I = len(O_G)   # task number
Q = len(O_GG)  # communication task number
Succ_OG = [[] for i in range(I)]  
Succ_OGG = [[] for q in range(Q)]
Succ_add_OG = [[] for i in range(I)]
Succ_add_OGG = [[] for q in range(Q)]
dirPred_OG = [[] for i in range(I)]
dirPred_add_OG = [[] for i in range(I)]
dirPred_OGG = []

# Succ_OG
for i in range(I - 1, -1, -1):
    Succ_OG[i].extend(dirSucc_OG[i])
    if Succ_OG[i] != []:
        for j in dirSucc_OG[i]:
            Succ_OG[i] = list(set(Succ_OG[i] + Succ_OG[O_G.index(j)]))
    Succ_OG[i].sort()
# Succ_add_OG
for i in range(I - 1, -1, -1):
    Succ_add_OG[i].extend(dirSucc_add_OG[i])
    if Succ_add_OG[i] != []:
        for j in dirSucc_add_OG[i]:
            if j in O_G:
                Succ_add_OG[i] = list(set(Succ_add_OG[i] + Succ_add_OG[O_G.index(j)]))
    Succ_add_OG[i].sort()
# Succ_OGG
for q in range(Q - 1, -1, -1):
    Succ_OGG[q].append(dirSucc_OGG[q])
    Succ_OGG[q] = list(set(Succ_OGG[q] + Succ_OG[O_G.index(dirSucc_OGG[q])]))
# Succ_add_OGG
for q in range(Q - 1, -1, -1):
    Succ_add_OGG[q].append(dirSucc_OGG[q])
    Succ_add_OGG[q] = list(set(Succ_add_OGG[q] + Succ_add_OG[O_G.index(dirSucc_OGG[q])]))
# dirPred_OG
for i in range(I):
    for j in dirSucc_OG[i]:
        dirPred_OG[O_G.index(j)].append(O_G[i])
# dirPred_add_OG
dirPred_add_OG = dirPred_OG
for q in range(Q):
    dirPred_add_OG[O_G.index(dirSucc_OGG[q])].append(O_GG[q])
# dirPred_OGG
for q in range(Q):
    for i in range(I):
        if O_GG[q] in dirSucc_add_OG[i]:
            dirPred_OGG.append(O_G[i])
            break
dirSucc_OGtoOGG = [[] for i in range(I)] 
dirPred_OGtoOGG = [[] for i in range(I)]  
for i in range(I):
    for j in dirSucc_add_OG[i]:
        if j in O_GG:
            dirSucc_OGtoOGG[i].append(j)
    for j in dirPred_add_OG[i]:
        if j in O_GG:
            dirPred_OGtoOGG[i].append(j)

m = Model('Moirai')

x = {}  # x_i
z = {}  # z_q
S = {}  # S_i
C = {}  # C_i
u = {}  # u_qk1k2
delta_ij = {}  # delta1_ij
delta_rq = {}  # delta_rq
for i in range(I):
    for k in range(K):
        x[i, k] = m.addVar(vtype=GRB.BINARY, name='x_%d_%d' % (i, k))
    for j in range(I):
        delta_ij[i, j] = m.addVar(vtype=GRB.BINARY, name='delta_ij_%d_%d' % (i, j))
for q in range(Q):
    z[q] = m.addVar(lb=0, vtype=GRB.BINARY, name='z_%d' % q)
    for r in range(Q):
        delta_rq[r, q] = m.addVar(vtype=GRB.BINARY, name='delta_rq_%d_%d' % (r, q))
    for k1 in range(K):
        for k2 in range(K):
            u[q, k1, k2] = m.addVar(lb=0, vtype=GRB.BINARY, name='u_%d_%d_%d' % (q, k1, k2))

for i in range(I + Q):
    S[i] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='S_%d' % i)
    C[i] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='C_%d' % i)
makespan = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='makespan')

m.update()
# objective
m.setObjective(makespan, GRB.MINIMIZE)

# constraints
# (1)C_i<=S_j
for i in range(I):
    for j in dirSucc_OGtoOGG[i]:
        m.addConstr(C[O_G[i]] <= S[j])
    for j in dirPred_OGtoOGG[i]:
        m.addConstr(S[O_G[i]] >= C[j])
    for j in Succ_OG[i]:
        m.addConstr(C[O_G[i]] <= S[j])
# (2)C_i=S_i+sum(p_ik*x_ik)
# (3)sum(x_ik)=1
for i in range(I):
    m.addConstr(quicksum(ptask[k][i] * x[i, k] for k in range(K)) + S[O_G[i]] == C[O_G[i]])
    m.addConstr(quicksum(x[i, k] for k in range(K)) == 1)
# (4)Memory constraints
for k in range(K):
    m.addConstr(quicksum(mtask[i] * x[i, k] for i in range(I)) <= Mem[k])
# (5)Non-overlapping constraints
for i in range(I):
    for j in range(I):
        if i != j and (O_G[i] not in Succ_OG[j]) and (O_G[j] not in Succ_OG[i]):
            for k in range(K):
                m.addConstr(S[O_G[i]] >= C[O_G[j]] - Ms * delta_ij[i, j] - Ml * (2 - x[i, k] - x[j, k]))
                m.addConstr(S[O_G[j]] >= C[O_G[i]] - Ms * (1 - delta_ij[i, j]) - Ml * (2 - x[i, k] - x[j, k]))
# (6)Communication constraints
for q in range(Q):
    i = dirPred_OGG[q]
    j = dirSucc_OGG[q]
    for k in range(K):
        m.addConstr(z[q] <= 2 - x[O_G.index(i), k] - x[O_G.index(j), k])
        m.addConstr(z[q] >= x[O_G.index(i), k] - x[O_G.index(j), k])
        m.addConstr(z[q] >= x[O_G.index(j), k] - x[O_G.index(i), k])
    m.addConstr(quicksum(quicksum(u[q, k1, k2] for k1 in range(K)) for k2 in range(K)) == z[q])
    for k1 in range(K):
        for k2 in range(K):
            if k1 != k2:
                m.addConstr(u[q, k1, k2] >= x[O_G.index(i), k1] + x[O_G.index(j), k2] - 1)
    m.addConstr(
        C[O_GG[q]] == S[O_GG[q]] + quicksum(quicksum(u[q, k1, k2] * pcomm[k1][k2][q] for k1 in range(K)) for k2 in range(K)))

# (7)Congestion constraints
for q in range(Q):
    for r in range(Q):
        if (q != r) and (q not in Succ_add_OGG[r]) and (r not in Succ_add_OGG[q]):
            a = dirPred_OGG[q]
            c = dirPred_OGG[r]
            b = dirSucc_OGG[q]
            d = dirSucc_OGG[r]
            for k in range(K):
                m.addConstr(S[O_GG[q]] >= C[O_GG[r]] - Ms * delta_rq[r, q] - Ml * (2 - z[q] - z[r]) + Mr * (
                        x[O_G.index(a), k] + x[O_G.index(c), k] - x[O_G.index(b), k] - x[O_G.index(d), k] - 2))
                m.addConstr(S[O_GG[r]] >= C[O_GG[q]] - Ms * (1 - delta_rq[r, q]) - Ml * (2 - z[q] - z[r]) + Mr * (
                        x[O_G.index(a), k] + x[O_G.index(c), k] - x[O_G.index(b), k] - x[O_G.index(d), k] - 2))
                m.addConstr(S[O_GG[q]] >= C[O_GG[r]] - Ms * delta_rq[r, q] - Ml * (2 - z[q] - z[r]) + Mr * (
                        x[O_G.index(b), k] + x[O_G.index(d), k] - x[O_G.index(a), k] - x[O_G.index(c), k] - 2))
                m.addConstr(S[O_GG[r]] >= C[O_GG[q]] - Ms * (1 - delta_rq[r, q]) - Ml * (2 - z[q] - z[r]) + Mr * (
                        x[O_G.index(b), k] + x[O_G.index(d), k] - x[O_G.index(a), k] - x[O_G.index(c), k] - 2))

# (8) objective function
for i in range(I + Q):
    m.addConstr(makespan >= C[i])

start = datetime.datetime.now()
# result
m.optimize()
a = m.getVars()
# output identifier
# for var in a:
#     print(f"{var.varName}: {round(var.X, 3)}")
end = datetime.datetime.now()
print(end-start)
