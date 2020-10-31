# -*- cofing: utfg- 8 -*-
'''
auther: Longxin
data: 2017/11/26
version: 2.0
description: 这是论文中的代码，相对于第一篇论文中的代码，我改动了一些约束条件，但是
在改动约束条件的同时也将论文的代码进行一遍重构，方便用于测试
chang_version: 3.0
change_author: Longxin
change_descritpion: 由于data的关系在全局中使用了，所有我们在计算的时候是错误，为了解决这个bug
有两种解决办法： 第一种解决办法：

change_version: 4.0
change_author: longxin
change_description: 对代码进行优化 增加online和offline两个版本
'''
import numpy as np
import math
import time
import xlwt
import xlrd
# from xlutils.copy import copy
import sys

'''
这是一个模型，定义模型中需要的最基本的结构，但是其中的求解算法我们将之看作模型
的一部分，将算法嵌入到模型当中。
param: W 任务载荷workload，大小为M*N，M为设备数量N为任务数量，大小表示为任务需要
的cpu执行周期
pre: 任务前趋图，表示第i个任务前面的任务，为[listpre[]]类型的数据结构
data: 每个任务的数据大小，size为M*N，表示在任务迁移过程中的传输的数据量的大小
W_cloud: 表示fog-cloud与cloud之间的带宽大小
W_d: 标识移动设备与fog-cloud之间的带宽大小
buget: 表示对于能耗的buget
'''

local_dir = r'C:\Users\derfei\Desktop\TMS_Exp\Mobile\Mobile\schedule_algor'
M = 1
N = 4

Workload = np.array([[10, 12, 8, 12]])
Datasize = np.array([[8, 4, 10, 2]])
k = math.pow(10, -11) # the effective switched capacitance depending on the chip architecture
pre = [[], [0], [0], [1,2]]
W_edge = 5  # the bandwith between edge
upload_rate_between_mobile_edge = None
W_cloud = 10  # the bandwith between cloud and edge.
Gain = 10 # the channel gain
Ptx = 0.2 # transmission power
you_mp = 1
edge_computing_capacity = 3.6
cloud_computing_capacity = 4
local_computing_capacity = 1
alpha_f = 1.6
beta_f = 1.6
e_for_energy = 1.2
alpha_c = 2.6
beta_c = 2.6
P0 = 0.1
Pe_p = 1
Pc_q = 2
Budget = 10
Device_num = 3
Max_k = 100000
Epsion = 0.01
Graph_depth = 3
Input_datasize = 10

def Ptx_load(N):
    ptx = []
    tmp = []
    for i in range(N):
        tmp.append(100)
    ptx.append(tmp)
    return ptx

# the formula 1
def computer_R_mnp():
    if upload_rate_between_mobile_edge == None:
        return W_edge * math.log2(1 + ((Ptx*Gain)/(you_mp**2)))
    else:
        return float(upload_rate_between_mobile_edge)

# formula 2
def computer_Tl_mn(W_mn):
    return W_mn/local_computing_capacity

# formula 3
def computer_El_mn(w_mn):
    import math
    return k*w_mn*(local_computing_capacity)*local_computing_capacity

# formula 4
def computer_Tf_mn(w_mn):
    return w_mn/edge_computing_capacity

#formula 5
def computer_Ef_mn(te_mn):
    import math
    return (alpha_f*(edge_computing_capacity/math.pow(10, 9))**e_for_energy + beta_f) * te_mn

#formula 6
def computer_Ec_mn(tc_mnq):
    return (alpha_c*(cloud_computing_capacity/math.pow(10, 9)**e_for_energy) + beta_c)*tc_mnq

#formula 7
def computer_Tc_mnq(w_mn):
    return w_mn/cloud_computing_capacity

# formula 8
def computer_Tt_mn(d_mn, R_mnp):
    return d_mn/R_mnp

# formula 9
def computer_Et_mn(Ptx_mnp, Tt_mn):
    return Ptx_mnp*Tt_mn

# formula 10 1000 the branwithdith between the remote server
def computer_Tr_mnp(d_mn):
    return d_mn/W_cloud

# formula 11
def computer_Es_mn(Tr_mnq):
    return P0*Tr_mnq

# formula 12 which means find the max value in the pre
def computer_TRl_mn(pre, TFl_m, TFf_m, TFc_m, Xc, Xf):
    tmp = [0]
    for i in pre:
        tmp.append(max(TFl_m[i]*(1-Xc[M-1][i])*(1-Xf[M-1][i]),
                       TFf_m[i]*Xf[M-1][i], TFc_m[i]*Xc[M-1][i]))
    return max(tmp)

# formula 14
def computer_TFl_mn(Tl_mn, TRl_mn):
    return Tl_mn + TRl_mn

# formula 15
def computer_TRf_mn(TFt_mn, TFf_m, TFc_m, pre, Xc, Xf):
    tmp = [0]
    tmp1 = [0]
    tmp2 = [0]
    for i in pre:
        tmp.append(Xf[M-1][i]*TFf_m[i])
        tmp1.append(Xc[M-1][i]*TFc_m[i])
        tmp2.append(TFt_mn*(1-Xf[M-1][i])*(1-Xc[M-1][i]))
    max_TFf = max(tmp)
    max_TFc = max(tmp1)

    '@warning: there may be some error'
    max_tfmn = max(tmp2)
    return max(max_tfmn, max_TFf, max_TFc)

# formula 16
def computer_TRc_mn(TFt_mn,  Tr_mn, TFc_m, TFr_mn, pre, Xc, Xf):
    tmp  = [0]
    tmpTFr = [0]
    tmpTFtplusTrmn = [0]
    for i in pre:
        tmp.append(TFc_m[i]*Xc[M-1][i])
        tmpTFr.append(TFr_mn*Xf[M-1][i])
        tmpTFtplusTrmn.append((TFt_mn+Tr_mn)*(1-Xc[M-1][i])*(1-Xf[M-1][i]))
    max_TFc = max(tmp)
    max_TFr = max(tmpTFr)
    max_TFTplusTrmn = max(tmpTFtplusTrmn)
    return max(max_TFTplusTrmn, max_TFc, max_TFr)

# formula 17
def computer_TFt_mn(Tt_mn, TFl_m, pre):
    tmp = [0]
    for i in pre:
        tmp.append(TFl_m[i])
    return Tt_mn+max(tmp)

# formula 18
def computer_TFf_mn(Tf_mn, TRf_mn):
    return Tf_mn + TRf_mn

# formula 19
def computer_TFc_mn(Tc_mnq, TRc_mn):
    return Tc_mnq + TRc_mn

# formula 20
def computer_TFr_mn(Tr_mn, TFf_m, pre):
    tmp = [0]
    for i in pre:
        tmp.append(TFf_m[i])
    return Tr_mn+max(tmp)

#  formula 23
def computer_Uf_p(Es, Xc, Ef, Xf, Datasize):
    tmp = 0.0
    for m in range(M):
        for n in range(N):
            tmp = tmp + Pe_p*Datasize[m][n]*Xf[m][n] - Es[n]*Xc[m][n] - Ef[n]*Xf[m][n]
    return tmp


# formula 24
def computer_Uc_q(d, Ec, Xc):
    tmp = 0
    for m in range(M):
        for n in range(N):
            tmp = tmp + Pc_q*d[m][n]*Xc[m][n] - Ec[n]*Xc[m][n]
    return tmp

# formula 26
def computer_C_mn(Xc_mn, El_mn, Et_mn, d_mn, Xf_mn):
    return (1 - Xc_mn)*(El_mn*(1-Xf_mn) + (Pe_p*d_mn + Et_mn)*Xf_mn) + (Pc_q*d_mn+Et_mn+computer_Es_mn(computer_Tr_mnp(d_mn)))*Xc_mn

# formula 27
def computer_T_m(TFl_m, TFf_m, TFc_m, Xfm, XCm):
    tmp = 0.0
    i = len(pre) -1
    tmp = (1-XCm[i])*(1-Xfm[i])*TFl_m[i] + Xfm[i]*TFf_m[i] + XCm[i]*TFc_m[i]
    return tmp

def model():
    import time

    Xf = np.zeros(shape=(M, N), dtype=np.float32)
    Xc = np.zeros(shape=(M, N), dtype=np.float32)
    R_mp = np.zeros(shape=N, dtype=np.float32)# size N
    Tl_m = np.zeros(shape=N, dtype=np.float32) # size N
    El_m = np.zeros(shape=N, dtype=np.float32) # size N
    Tf_m = np.zeros(shape=N, dtype=np.float32) # size N
    Ef_m = np.zeros(shape=N, dtype=np.float32) # size N
    Ec_m = np.zeros(shape=N, dtype=np.float32) # size N
    Tc_mq = np.zeros(shape=N, dtype=np.float32) # size N
    Tt_m = np.zeros(shape=N, dtype=np.float32) # size N
    Et_m = np.zeros(shape=N, dtype=np.float32) # size N
    Tr_mq = np.zeros(shape=N, dtype=np.float32) # size N
    Es_m = np.zeros(shape=N, dtype=np.float32) # size N
    TRl_m = np.zeros(shape=N, dtype=np.float32) # size N
    TFl_m = np.zeros(shape=N, dtype=np.float32) # size N
    TRf_m = np.zeros(shape=N, dtype=np.float32) # size N
    TRc_m = np.zeros(shape=N, dtype=np.float32) # size N
    TFt_m = np.zeros(shape=N, dtype=np.float32) # size N
    TFf_m = np.zeros(shape=N, dtype=np.float32) # size N
    TFc_m = np.zeros(shape=N, dtype=np.float32) # size N
    TFr_m = np.zeros(shape=N, dtype=np.float32) # size N
    C_m = np.zeros(shape=N, dtype=np.float32) # size N

    for n in range(N):
        # init all the paraeters
        R_mp[n] = computer_R_mnp() # formula one
        Tl_m[n] = computer_Tl_mn(Workload[M-1][n])
        El_m[n] = computer_El_mn(Workload[M-1][n])

        Tf_m[n] = computer_Tf_mn(Workload[M-1][n])
        Ef_m[n] = computer_Ef_mn(Tf_m[n])

        Tc_mq[n] = computer_Tc_mnq(Workload[M-1][n])
        Ec_m[n] = computer_Ec_mn(Tc_mq[n])

        Tt_m[n] = computer_Tt_mn(Datasize[M-1][n], R_mp[n])
        Et_m[n] = computer_Et_mn(Ptx, Tt_m[n])

        Tr_mq[n] = computer_Tr_mnp(Datasize[M-1][n])
        Es_m[n] = computer_Es_mn(Tr_mnq=Tr_mq[n])
        #
        # if len(pre[n])==0:
        #     # the first subtask is assigned to be executed on the local device,
        #     # so init zero to all params except the TFl
        #     TRl_m[n] = 0
        #     TRf_m[n] = 0
        #     TRc_m[n] = 0
        #
        #     TFl_m[n] = computer_TFl_mn(Tl_mn=Tl_m[n], TRl_mn=TRl_m[n])
        #     TFt_m[n] = 0
        #     TFc_m[n] = 0
        #     TFf_m[n] = 0
        # else:
        # computer the ready time for the local computing
        TRl_m[n] = computer_TRl_mn(pre[n],TFl_m, TFf_m, TFc_m, Xc, Xf)

        # computer the ready time for the edge computings
        TFt_m[n] = computer_TFt_mn(Tt_mn=Tt_m[n], TFl_m=TFl_m, pre=pre[n])
        TRf_m[n] = computer_TRf_mn(TFt_mn=TFt_m[n], TFf_m=TFf_m, TFc_m=TFc_m, pre=pre[n],
                                   Xc=Xc, Xf=Xf)

        # computer the ready time for the cloud computing
        TFr_m[n] = computer_TFr_mn(Tr_mn=Tr_mq[n], TFf_m=TFf_m, pre=pre[n])
        TRc_m[n] = computer_TRc_mn(TFt_mn=TFt_m[n], Tr_mn=Tr_mq[n], TFc_m=TFc_m, TFr_mn=TFr_m[n], pre=pre[n],
                                   Xc=Xc, Xf=Xf)
        # computer the task finish time
        TFl_m[n] = computer_TFl_mn(Tl_m[n], TRl_m[n])
        TFf_m[n] = computer_TFf_mn(Tf_mn=Tf_m[n], TRf_mn=TRf_m[n])
        TFc_m[n] = computer_TFc_mn(Tc_mnq=Tc_mq[n], TRc_mn=TRc_m[n])

        if min(TFl_m[n], TFf_m[n], TFc_m[n])==TFl_m[n]:
            Xf[M-1][n] = 0.0
            Xc[M-1][n] = 0.0

            # reset the task finish time
            TFf_m[n] = 0
            TFc_m[n] = 0

        else:
            if min(TFl_m[n], TFf_m[n], TFc_m[n])==TFc_m[n]:
                Xf[M - 1][n] = 0.0
                Xc[M - 1][n] = 1.0

                # reset the task finish time
                TFl_m[n] = 0
                TFf_m[n] = 0
            else:
                Xf[M - 1][n] = 1.0
                Xc[M - 1][n] = 0.0

                # reset the task finish time
                TFl_m[n] = 0
                TFc_m[n] = 0
        # computer the total comusumption
        C_m[n] = computer_C_mn(Xc_mn=Xc[M - 1][n], El_mn=El_m[n], d_mn=Datasize[M - 1][n],
                               Xf_mn=Xf[M - 1][n], Et_mn=Et_m[n])

    # computer the utility of edge and cloud
    Uf_p = computer_Uf_p(Es_m, Xc, Ef_m, Xf, Datasize)
    Uc_q = computer_Uc_q(d=Datasize, Ec=Ec_m, Xc=Xc)


    # the principle of adjusting the comsumption in a greedy way:
    # the perfect way: choose the one which increase the minimize time while with the decrease of comsumption
    # the convenient way: choose the minimize data subtask from the cloud to edge
    change_count = 0
    while sum(C_m) > Budget:
        if sum(Xc[M-1]) != 0:
            tmpEsm = Es_m
            tmpEsm[Xc[0, :] == 0] = sys.maxsize
            # y = np.where(tmpEsm == np.min(tmpEsm))[0][0]
            y = np.where(tmpEsm !=  float(sys.maxsize))[0][-1]
            Xf[M-1][y] = 1
            Xc[M-1][y] = 0
        else:
            tmpEtm = Et_m
            tmpEtm[Xf[0, :] == 0] = sys.maxsize
            # y = np.where(tmpEtm == np.min(tmpEtm))[0][0]
            y = np.where(tmpEtm != float(sys.maxsize))[0][-1]
            Xf[M-1][y] = 0
            Xc[M-1][y] = 0

        # update the cm
        for i in range(N):
            # computer the total comusumption
            C_m[i] = computer_C_mn(Xc_mn=Xc[M - 1][i], El_mn=El_m[i], d_mn=Datasize[M - 1][i],
                                   Xf_mn=Xf[M - 1][i], Et_mn=Et_m[i])

        change_count += 1

        if change_count >= 2*N:
            break

        # print("Decrease the cost: {0} Xf {1} Xc {2}".format(sum(C_m), Xf, Xc))

    # the priinciple of adjusting the utility in a greedy way:
    # the perfect way: choose the one which increase the total time minimize with the decrease of utility
    # the convenient way: choose the minimize data subtask from the cloud to edge
    Uf_p = computer_Uf_p(Es_m, Xc, Ef_m, Xf, Datasize)

    change_count = 0
    while Uf_p < 0:
        Es_m_dive_Ee_m = np.array([((Es_m[i]+Pe_p*Datasize[M-1][i])/(Ef_m[i]+Es_m[i]))*Xc[M-1][i] for i in range(N)])
        if sum(Xc[M-1]) != 0 and len(np.where(Es_m_dive_Ee_m > 1)[0]) != 0:
            # Es_m_dive_Ee_m = Es_m_dive_Ee_m[np.where(Es_m_dive_Ee_m > 1)]
            # max_value = np.max(Es_m_dive_Ee_m)
            # u = Es_m_dive_Ee_m.index(max_value)
            # u = np.argwhere(Es_m_dive_Ee_m == max_value)[0]
            u = np.where(Es_m_dive_Ee_m > 1)[0][-1]
            Xf[M-1][u] = 1
            Xc[M-1][u] = 0
        else:
            Pefdmn_div_Eemn = [(Pe_p*Datasize[M-1][i]+Et_m[i])/Ef_m[i] for i in range(N)]
            min_value = np.min(Pefdmn_div_Eemn)
            u = Pefdmn_div_Eemn.index(min_value)
            Xf[M-1][u] = 0
            Xc[M-1][u] = 0

        # update the utility
        Uf_p = computer_Uf_p(Es_m, Xc, Ef_m, Xf, Datasize)

        change_count += 1

        if change_count >= N:
            break

    'feature: recomputer the cost task finish time'
    # print("The xf and Xc is {0} {1} The sum cm is: {2} The ufp is {3}".format(Xf, Xc, sum(C_m), Uf_p))

    # update the cm
    for i in range(N):
        # computer the total comusumption
        C_m[i] = computer_C_mn(Xc_mn=Xc[M - 1][i], El_mn=El_m[i], d_mn=Datasize[M - 1][i],
                               Xf_mn=Xf[M - 1][i], Et_mn=Et_m[i])

    tmp = sum(C_m)
    if Uf_p < 0 or sum(C_m) > Budget:
        return -1, -1

    return Xf, Xc

def get_nexttask_list(pre):
    '''
    get the next task list by pre
    :param pre:  formertask list
    :return: next task list
    '''
    'change the formertask list to the next task list'
    nexttask_list = {}
    for i, tmp in enumerate(pre):
        if len(pre) != 0:
            for tmptask in tmp:
                if tmptask in nexttask_list.keys():
                    nexttask_list.get(tmptask).append(i)
                else:
                    nexttask_list[tmptask] = []
                    nexttask_list.get(tmptask).append(i)

    for i in range(N):
        if i not in nexttask_list.keys():
            nexttask_list[i] = []


    'set the last task'
    nexttask_list[len(pre) - 1] = []
    return nexttask_list

def bfs(pre):
    '''
    use the bfs to get the visited sequence
    :param nexttasklist: formertask list
    :return: vist sequence list
    '''
    from queue import Queue
    'visited sequence'
    visite_sequence = []

    nexttask_list = get_nexttask_list(pre)
    visited = set()
    visited_queue = Queue()

    visited_queue.put(0)
    visited.add(0)

    while not visited_queue.empty():
        v = visited_queue.get()
        visite_sequence.append(v)

        tmp_nexttask_list = nexttask_list[v]

        for tmp in tmp_nexttask_list:
            if tmp not in visited:
                visited_queue.put(tmp)
                visited.add(tmp)

    return visite_sequence

def find_maxmun_latency():
    import numpy as np

    L = np.zeros(shape=(N, Device_num))
    visited_sequence = bfs(pre)

    for i in visited_sequence:
        'if i is  a leaf'
        if len(pre[i]) == 0:
            for j in range(Device_num):
                'Finish time on the mobile device'
                if j == 0:
                    L[i][j] = computer_Tl_mn(Workload[M-1][i])
                if j == 1:
                    L[i][j] = computer_Tf_mn(Workload[M-1][i])

                if j == 2:
                    L[i][j] = computer_Tc_mnq(Workload[M-1][i])

        else:
            for j in range(Device_num):
                'choose the max latency for the formertask'
                max_latency_former_j = 0
                for tmp in pre[i]:
                    for k in range(Device_num):
                        'if the formertask is the same executed place'
                        if k == j:
                            max_latency_former_j = max(max_latency_former_j, L[tmp][k])
                        else:
                            # @warmming: take the download communication into condideration or just condiderat the upload consideartion
                            if j == 0 and k == 1:
                                max_latency_former_j = max(max_latency_former_j, L[tmp][k]+computer_Tt_mn(Datasize[M-1][tmp], R_mnp=computer_R_mnp()))
                            elif j == 1 and k == 2:
                                max_latency_former_j = max(max_latency_former_j, L[tmp][k]+computer_Tr_mnp(Datasize[M-1][tmp]))
                            else:
                                max_latency_former_j = max(max_latency_former_j, L[tmp][k])

                if j == 0:
                    L[i][j] = computer_Tl_mn(Workload[M-1][i])
                if j == 1:
                    L[i][j] = computer_Tf_mn(Workload[M-1][i])

                if j == 2:
                    L[i][j] = computer_Tc_mnq(Workload[M-1][i])

                L[i][j] += max_latency_former_j

    return max(L[len(pre)-1])

def quantitu(a, b):
    return int(a/b) + 1

def computer_Tij(i,j):
    Tij = 0
    if j == 0:
        Tij = computer_Tl_mn(Workload[M-1][i])
    if j == 1:
        Tij = computer_Tf_mn(Workload[M-1][i])

    if j == 2:
        Tij = computer_Tc_mnq(Workload[M-1][i])

    return Tij

def computer_Cij(i,j):
    Cij = 0
    if j == 0:
        Cij = computer_El_mn(Workload[M-1][i])
    if j == 1:
        Cij = Pe_p*Datasize[M-1][i]
    if j == 2:
        Cij = Pc_q*Datasize[M-1][i]

    return Cij

def computer_Uij(i, j):
    Uij = 0
    if j == 0:
        Uij = 0
    if j == 1:
        Uij = Pe_p*Datasize[M-1][i] - computer_Ef_mn(computer_Tf_mn(Workload[M-1][i]))
    if j == 2:
        Uij = -computer_Es_mn(computer_Tr_mnp(Datasize[M-1][i]))

    return Uij

def dp(q, Tr, O):
    import numpy as np
    import sys

    global  Max_k

    K = quantitu(Tr, O)
    Max_k = K
    # print("The size of K is: ", K)

    'record for the best solution of minimum latency'
    C = np.zeros(shape=(N, Device_num, Max_k))

    for i in q:
        if len(pre[i]) == 0:
            for j in range(Device_num):
                if j == 0:
                    Tij = computer_Tij(i, j)
                    for k in range(K):
                        if k >= quantitu(Tij, O):
                            C[i,j,k] = computer_Cij(i,j)
                        else:
                            C[i,j,k] = sys.maxsize
                elif j == 1:
                    Tij = computer_Tij(i, j)
                    for k in range(K):
                        if k >= quantitu(Tij+computer_Tt_mn(Input_datasize, computer_R_mnp()), O):
                            C[i, j, k] = computer_Cij(i, j) + computer_Et_mn(Ptx, computer_Tt_mn(Input_datasize, computer_R_mnp()))
                        else:
                            C[i, j, k] = sys.maxsize
                else:
                    Tij = computer_Tij(i, j)
                    for k in range(K):
                        if k >= quantitu(Tij+computer_Tr_mnp(Input_datasize)+computer_Tt_mn(Input_datasize, computer_R_mnp()), O):
                            C[i, j, k] = computer_Cij(i, j) + computer_Es_mn(computer_Tr_mnp(Input_datasize)) + \
                                         computer_Et_mn(Ptx, computer_Tt_mn(Input_datasize, computer_R_mnp()))
                        else:
                            C[i, j, k] = sys.maxsize

            # print("C[0, :, :] ", C[0, :, :])
        else:
            for j in range(Device_num):
                for k in range(K):
                    Cij = computer_Cij(i,j)
                    Tij = computer_Tij(i,j)

                    C[i,j,k] = Cij
                    sum_child_cost = 0
                    'find the minimize cost for the chile node'
                    for tmp in pre[i]:
                        min_cost = sys.maxsize

                        'for each device, choose the best solution'
                        for xm in range(Device_num):
                            'computer the km'
                            Tmi = 0
                            Costmj = 0
                            if xm == 0 and j == 1:
                                Tmi = computer_Tt_mn(Datasize[M-1][tmp], computer_R_mnp())
                                Costmj = computer_Et_mn(Ptx, Tmi)

                            if xm == 1 and j == 2:
                                Tmi = computer_Tr_mnp(Datasize[M-1][tmp])
                                Costmj = computer_Es_mn(Tmi)

                            if xm == 0 and j == 2:
                                Tmi = computer_Tr_mnp(Datasize[M-1][tmp]) + computer_Tt_mn(Datasize[M-1][tmp], computer_R_mnp())
                                Costmj = computer_Es_mn(computer_Tr_mnp(Datasize[M-1][tmp])) + computer_Et_mn(Ptx, computer_Tt_mn(Datasize[M-1][tmp],
                                                                                                  computer_R_mnp()))


                            Km = quantitu((Tij+Tmi), O)
                            # print("The km is: {0} The Tij + Tmi = {1}".format(Km, Tij+Tmi))

                            if k - Km < 0:
                                min_cost = sys.maxsize
                            else:
                                if C[tmp,xm,k-Km] == sys.maxsize:
                                    continue
                                else:
                                    min_cost = min(min_cost, (C[tmp,xm,k-Km]+Costmj))


                        if min_cost == sys.maxsize:
                            sum_child_cost = sys.maxsize
                        else:
                            sum_child_cost += min_cost
                    C[i,j,k] += sum_child_cost
    'Choose the minimize delay for with energy cost budget'
    min_j = -1
    min_k = Max_k
    # print("The min of  C[N-1, j, k] is: ", np.min(C[N-1, :, :]))
    for j in range(Device_num):
        for k in range(Max_k):
            if C[N-1, j, k] <= Budget and k < min_k:

                # 'choose the policy fit the budget for the utility'
                # X = get_offloading_policy(C, j, k, O)

                'computer the utility of the policy'
                # u = 0

                # for tmp in range(len(pre)):
                #     u += computer_Uij(tmp, X[M - 1][tmp] - 1)
                # if u >= 0:
                    # print("The final offloading policy is: ", X)
                min_k = k
                min_j = j

    # print("The last Cm is: ", C[N-1, :, :])
    if min_k == Max_k:
        return -1, -1, -1, -1
    return min_k, C[N-1, min_j, min_k], C, min_j


def ftpas(l):
    '''
    ftpas algorithm
    :param l:  depth of the task graph
    :return:
    '''
    T = find_maxmun_latency()*10
    q = bfs(pre)

    best_x = sys.maxsize
    best_or = sys.maxsize
    best_C = None
    last_j = -1

    for r in range(1, int(math.log2(T))):
        Tr = T / (2**(r-1))
        Or = (Epsion*T) / (l*(2**r))

        x,min_cost,C, min_j = dp(q, Tr, Or)
        # print("The x is {0} Tr: {1} Or: {2} Min cost: {3}".format(x, Tr, Or, min_cost))


        if x != -1 and x*Or < best_x:
            best_x = x*Or
            best_or = Or
            best_C = C
            last_j = min_j
        # break
    return best_x,best_or, best_C, last_j

def test_getnexttask_list():
    print("The next task list is: ", get_nexttask_list(pre))

def test_bfs():
    print("The bdfs of the task graph is: ", bfs(pre))


def test_find_maxmun_latency():
    print("The max mum latency is: ", find_maxmun_latency())


def test_ftpas():
    print("The ftpas is : ", ftpas(Graph_depth))


def test_get_offloading_policy():
    best_x, best_or, C, last_j = ftpas(Graph_depth)

    'get the best K'
    best_k = int(best_x/best_or)

    'get the last task j'
    X = get_offloading_policy(C, last_j, best_k, best_or)

    # print("The last offloading policy: ", X)

def test_model():
    print("The model is: ", model())

def get_time_cost(Xf, Xc):
    R_mp = np.zeros(shape=N, dtype=np.float32)  # size N
    Tl_m = np.zeros(shape=N, dtype=np.float32)  # size N
    El_m = np.zeros(shape=N, dtype=np.float32)  # size N
    Tf_m = np.zeros(shape=N, dtype=np.float32)  # size N
    Ef_m = np.zeros(shape=N, dtype=np.float32)  # size N
    Ec_m = np.zeros(shape=N, dtype=np.float32)  # size N
    Tc_mq = np.zeros(shape=N, dtype=np.float32)  # size N
    Tt_m = np.zeros(shape=N, dtype=np.float32)  # size N
    Et_m = np.zeros(shape=N, dtype=np.float32)  # size N
    Tr_mq = np.zeros(shape=N, dtype=np.float32)  # size N
    Es_m = np.zeros(shape=N, dtype=np.float32)  # size N
    TRl_m = np.zeros(shape=N, dtype=np.float32)  # size N
    TFl_m = np.zeros(shape=N, dtype=np.float32)  # size N
    TRf_m = np.zeros(shape=N, dtype=np.float32)  # size N
    TRc_m = np.zeros(shape=N, dtype=np.float32)  # size N
    TFt_m = np.zeros(shape=N, dtype=np.float32)  # size N
    TFf_m = np.zeros(shape=N, dtype=np.float32)  # size N
    TFc_m = np.zeros(shape=N, dtype=np.float32)  # size N
    TFr_m = np.zeros(shape=N, dtype=np.float32)  # size N
    C_m = np.zeros(shape=N, dtype=np.float32)  # size N

    for n in range(N):
        # init all the paraeters
        R_mp[n] = computer_R_mnp()  # formula one
        Tl_m[n] = computer_Tl_mn(Workload[M - 1][n])
        El_m[n] = computer_El_mn(Workload[M - 1][n])

        Tf_m[n] = computer_Tf_mn(Workload[M - 1][n])
        Ef_m[n] = computer_Ef_mn(Tf_m[n])

        Tc_mq[n] = computer_Tc_mnq(Workload[M - 1][n])
        Ec_m[n] = computer_Ec_mn(Tc_mq[n])

        Tt_m[n] = computer_Tt_mn(Datasize[M - 1][n], R_mp[n])
        Et_m[n] = computer_Et_mn(Ptx, Tt_m[n])

        Tr_mq[n] = computer_Tr_mnp(Datasize[M - 1][n])
        Es_m[n] = computer_Es_mn(Tr_mnq=Tr_mq[n])

        # if len(pre[n]) == 0:
        #     # the first subtask is assigned to be executed on the local device,
        #     # so init zero to all params except the TFl
        #     TRl_m[n] = 0
        #     TRf_m[n] = 0
        #     TRc_m[n] = 0
        #
        #     TFl_m[n] = computer_TFl_mn(Tl_mn=Tl_m[n], TRl_mn=TRl_m[n])
        #     TFt_m[n] = 0
        #     TFc_m[n] = 0
        #     TFf_m[n] = 0
        # else:
        # computer the ready time for the local computing
        TRl_m[n] = computer_TRl_mn(pre[n], TFl_m, TFf_m, TFc_m, Xc, Xf)

        # computer the ready time for the edge computings
        TFt_m[n] = computer_TFt_mn(Tt_mn=Tt_m[n], TFl_m=TFl_m, pre=pre[n])
        TRf_m[n] = computer_TRf_mn(TFt_mn=TFt_m[n], TFf_m=TFf_m, TFc_m=TFc_m, pre=pre[n], Xc=Xc, Xf=Xf)

        # computer the ready time for the cloud computing
        TFr_m[n] = computer_TFr_mn(Tr_mn=Tr_mq[n], TFf_m=TFf_m, pre=pre[n])
        TRc_m[n] = computer_TRc_mn(TFt_mn=TFt_m[n], Tr_mn=Tr_mq[n], TFc_m=TFc_m, TFr_mn=TFr_m[n], pre=pre[n],
                                   Xc=Xc, Xf=Xf)



        # computer the task finish time
        TFl_m[n] = computer_TFl_mn(Tl_m[n], TRl_m[n])
        TFf_m[n] = computer_TFf_mn(Tf_mn=Tf_m[n], TRf_mn=TRf_m[n])
        TFc_m[n] = computer_TFc_mn(Tc_mnq=Tc_mq[n], TRc_mn=TRc_m[n])

        if (1-Xf[M-1][n])*(1-Xc[M-1][n]) == 1:

            # reset the task finish time
            TFf_m[n] = 0
            TFc_m[n] = 0
        else:
            if Xf[M-1][n] == 1:
                # reset the task finish time
                TFl_m[n] = 0
                TFc_m[n] = 0
            else:
                # reset the task finish time
                TFl_m[n] = 0
                TFf_m[n] = 0
        # computer the total comusumption
        C_m[n] = computer_C_mn(Xc_mn=Xc[M - 1][n], El_mn=El_m[n], d_mn=Datasize[M - 1][n],
                               Xf_mn=Xf[M - 1][n], Et_mn=Et_m[n])

    Uf_p = computer_Uf_p(Es_m, Xc, Ef_m, Xf, Datasize)
    Uc_q = computer_Uc_q(d=Datasize, Ec=Ec_m, Xc=Xc)

    'computer the Tm'
    Tm = np.zeros(shape=(N), dtype=np.float32)
    for i in range(N):
        Tm[i] = computer_T_m(TFl_m, TFf_m, TFc_m, Xf[M-1], Xc[M-1])

    return Tm[len(pre)-1], sum(C_m), Uf_p, Uc_q

def tranform_policy(X):
    import numpy as np

    Xm = np.zeros(shape=(M, N), dtype=np.float32)
    Xf = np.zeros(shape=(M, N), dtype=np.float32)
    Xc = np.zeros(shape=(M, N), dtype=np.float32)

    for i in range(N):
        if X[M-1][i] == 1:
            Xm[M-1][i] = 1
        if X[M-1][i] == 2:
            Xf[M-1][i] = 1
        if X[M-1][i] == 3:
            Xc[M-1][i] = 1

    return Xm, Xf, Xc

def get_utility(Xm, Xf, Xc):
    pass

def test_get_time_cost():
    Xf, Xc = model()
    if type(Xf) != int:
        time, cost = get_time_cost(Xf, Xc)
    else:
        time, cost = -1, -1
    # print("The SEG: Get the time {0} Get the cost {1}".format(time, cost))

    best_x, best_or, C, last_j = ftpas(Graph_depth)

    'get the best K'
    best_k = int(best_x / best_or)

    # print("Hermes get  the best time: ", best_x)
    # print("The Graph depth is: {0}".format(Graph_depth))

    herms_time, herms_cost = -1, -1
    if best_x != sys.maxsize:
        'get the last task j'
        X = get_offloading_policy(C, last_j, best_k, best_or)
        Xm, Xf, Xc = tranform_policy(X)
        herms_time, herms_cost = get_time_cost(Xf, Xc)
        # print("The offloading policy : Xf {0} Xc {1} time".format(Xf, Xc))
        # print("The hermers: Get the time {0} Get the cost {1}".format(herms_time, herms_cost))
    return time, cost, herms_time, herms_cost

def get_Xf_Xc():
    # print("Budget is: ", Budget)
    Xf_sep, Xc_sep = model()
    if type(Xf_sep) != int:
        time, cost, _, _, = get_time_cost(Xf_sep, Xc_sep)
    else:
        time, cost, _, _, = -1, -1
    # print("The SEG: Get the time {0} Get the cost {1}".format(time, cost))

    best_x, best_or, C, last_j = ftpas(Graph_depth)

    'get the best K'
    best_k = int(best_x / best_or)

    # print("Hermes get  the best time: ", best_x)
    # print("The Graph depth is: {0}".format(Graph_depth))

    herms_time, herms_cost = -1, -1
    Xf, Xc = -1, -1
    if best_x != sys.maxsize:
        'get the last task j'
        X = get_offloading_policy(C, last_j, best_k, best_or)
        Xm, Xf, Xc = tranform_policy(X)
        herms_time, herms_cost, _, _, = get_time_cost(Xf, Xc)
        # print("The offloading policy : Xf {0} Xc {1} time".format(Xf, Xc))
        # print("The hermers: Get the time {0} Get the cost {1}".format(herms_time, herms_cost))
    return Xf_sep, Xc_sep, Xf, Xc

def get_utility():
    import time
    start_time = time.time()
    Xf_sep, Xc_sep = model()
    sep_utility_e, sep_utility_c = 0, 0
    if type(Xf_sep) != int:
        _, cost, sep_utility_e, sep_utility_c, = get_time_cost(Xf_sep, Xc_sep)
    else:
        _, cost, sep_utility_e, sep_utility_c, = -1, -1, -1, -1
    # print("The SEG: Get the time {0} Get the cost {1}".format(time, cost))
    print("Sep get Xf {0} Sep get Xc {1}".format(sum(Xf_sep[0]), sum(Xc_sep[0])))
    sep_runningtime = time.time()-start_time

    start_time = time.time()

    best_x, best_or, C, last_j = ftpas(Graph_depth)

    'get the best K'
    best_k = int(best_x / best_or)

    # print("Hermes get  the best time: ", best_x)
    # print("The Graph depth is: {0}".format(Graph_depth))

    herms_time, herms_cost = -1, -1
    Xf, Xc = -1, -1
    if best_x != sys.maxsize:
        'get the last task j'
        X = get_offloading_policy(C, last_j, best_k, best_or)
        Xm, Xf, Xc = tranform_policy(X)
        herms_time, herms_cost, hermes_utility_e, hermes_utility_c, = get_time_cost(Xf, Xc)
        print("Hermes get ")
        # print("The offloading policy : Xf {0} Xc {1} time".format(Xf, Xc))
        # print("The hermers: Get the time {0} Get the cost {1}".format(herms_time, herms_cost))
    hemers_running_time = time.time()-start_time
    print("The sep utility cloud is {0} edge is {1} hermes utility edge is {2} hermes utility cloud is {3}".format(sep_utility_e, sep_utility_c, hermes_utility_e,
                                                                                                                   hermes_utility_c))
    start_time = time.time()
    bue, buc,Xf_bf, Xc_bf = brute_force()
    print("BruteForce get Xf {0} Sep get Xc {1}".format(sum(Xf_bf[0]), sum(Xc_bf[0])))
    bf_running_time = time.time()-start_time
    return  sep_utility_e, sep_utility_c, hermes_utility_e, hermes_utility_c, bue, buc, sep_runningtime, hemers_running_time, bf_running_time


def brute_force():
    start_index = [0 for i in range(N)]

    time_list = []
    Xf_list = []
    Xc_list = []
    Ue_list = []
    Uc_list = []

    while True:
        if start_index[0] != 2:
            start_index[0] += 1
        else:
            tmpindex = -1
            for i in range(1, N):
                if start_index[i] != 2:
                    tmpindex = i
                    break
            if tmpindex == -1:
                break
            else:
                start_index[tmpindex] += 1

        # 将tmpindex转换成为Xf,Xc
        Xf = np.zeros(shape=(M, N), dtype=np.float32)
        Xc = np.zeros(shape=(M, N), dtype=np.float32)
        for i in range(N):
            if start_index[i] == 1:
                Xf[0][i] = 1
            if start_index[i] == 2:
                Xc[0][i] = 1
        time, cost, ue, ec = get_time_cost(Xf, Xc)

        if cost < Budget and ue >= 0:
            time_list.append(time)
            Xf_list.append(Xf.copy())
            Xc_list.append(Xc.copy())
            Ue_list.append(ue)
            Uc_list.append(ec)
    # 获得最小的time
    min_time = min(time_list)
    min_index = time_list.index(min_time)
    print("暴力法获得最小时间: {0}  Ue is {1} Uc is {2}".format(min_time, Ue_list[min_index], Uc_list[min_index]))

    return Ue_list[min_index], Uc_list[min_index], Xf_list[min_index], Xf_list[min_index]




def get_Xf_Xc_sep():
    # print("Budget is: ", Budget)
    Xf_sep, Xc_sep = model()
    return Xf_sep, Xc_sep

def get_Xf_Xc_hermes():
    best_x, best_or, C, last_j = ftpas(Graph_depth)

    'get the best K'
    best_k = int(best_x / best_or)

    # print("Hermes get  the best time: ", best_x)
    # print("The Graph depth is: {0}".format(Graph_depth))

    herms_time, herms_cost = -1, -1
    Xf, Xc = -1, -1
    if best_x != sys.maxsize:
        'get the last task j'
        X = get_offloading_policy(C, last_j, best_k, best_or)
        Xm, Xf, Xc = tranform_policy(X)
        herms_time, herms_cost = get_time_cost(Xf, Xc)
        # print("The offloading policy : Xf {0} Xc {1} time".format(Xf, Xc))
        # print("The hermers: Get the time {0} Get the cost {1}".format(herms_time, herms_cost))
    return Xf, Xc

def get_offloading_policy(C, j, k, O):
    '''
    find the offloading policy
    :param C: cost array
    :param j: last task execute device
    :param k: mini energy cost
    :param O: The quatutize funciont
    :param q: visited sequence
    :return: offloading policy
    '''
    import numpy as np
    import sys

    if j == -1:
        # print("Can not find a answer")
        return

    X = np.zeros(shape=(M, N))

    q = bfs(pre)
    nexttask_list = get_nexttask_list(pre)
    # former_k = k
    former_k_dict = {}
    former_k_dict[N-1] = k

    for i in q[::-1]:
        if i == N-1:
            X[M-1][i] = j+1
        else:

            if X[M-1][i] != 0:
                continue
            'choose the minized index based on the former k'
            min_cost = sys.maxsize
            min_xm = -1

            'get the nexttask executed device'
            # if len(nexttask_list[i]) == 0:
            #     continue
            nexttask_i = nexttask_list[i][-1]
            nexttask_j = X[M-1][nexttask_i]-1

            'get forrmer k'
            former_k = former_k_dict[nexttask_i]

            for xm in range(Device_num):
                Cxmi = 0
                Tij = computer_Tij(nexttask_i, nexttask_j)
                Txmj = 0
                if xm == 0 and nexttask_j == 1:
                    Txmj = computer_Tt_mn(Datasize[M-1][i], computer_R_mnp())
                    Cxmi += computer_Et_mn(Ptx, Txmj)

                if xm == 0 and nexttask_j == 2:
                    Txmj = computer_Tr_mnp(Datasize[M-1][i]) + computer_Tt_mn(Datasize[M-1][i], computer_R_mnp())
                    Cxmi += computer_Et_mn(Ptx, computer_Tt_mn(Datasize[M-1][i], computer_R_mnp())) + computer_Es_mn(computer_Tr_mnp(Datasize[M-1][i]))


                if xm == 1 and nexttask_j == 2:
                    Txmj = computer_Tr_mnp(Datasize[M-1][i])
                    Cxmi += computer_Es_mn(computer_Tr_mnp(Datasize[M-1][i]))

                'computer Km and compare teh minimize cost'
                Km = quantitu(Tij+Txmj, O)

                if former_k - Km < 0:
                    continue

                if C[i, xm, former_k-Km] != sys.maxsize and C[i, xm, former_k-Km]+Cxmi < min_cost:
                    min_xm = xm
                    min_cost = C[i, xm, former_k-Km]+Cxmi

            'reset the former k'
            Tij = computer_Tij(nexttask_i, nexttask_j)
            Txmj = 0
            if min_xm == 0 and nexttask_j == 1:
                Txmj = computer_Tt_mn(Datasize[M-1][i], computer_R_mnp())

            if min_xm == 1 and nexttask_j == 2:
                Txmj = computer_Tr_mnp(Datasize[M-1][i])
            if min_xm == 0 and nexttask_j == 2:
                Txmj = computer_Tr_mnp(Datasize[M - 1][i]) + computer_Tt_mn(Datasize[M - 1][i], computer_R_mnp())

            former_k_dict[i] = former_k - quantitu(Tij+Txmj, O)

            'set the policy'
            X[M-1][i] = min_xm+1

    return X

def setTaskGraphGSep(graph_type):
    import numpy as np

    global Graph_depth
    global N
    global  Workload
    global  Datasize
    global Input_datasize
    global  pre
    import os
    import sys

    if graph_type == 1:
        pre = [[i - 1] for i in range(8)]
        pre[0] = []
        data_size = np.loadtxt(os.path.join(local_dir, 'vgg16_datasize.txt'))
        data_size[0][0] = 224 * 224 * 3
        data_size = data_size / (1024 * 1024)
        workload = '0.122599874	0.161675558	0.192462711	0.178823042	0.066766434	0.030215459 0.161675558 0.161675558'.split()
        workload = [float(tmp) for tmp in workload]
        workload = [tmp * 4.0 for tmp in workload]

        Datasize = [224*224*3, 112*112*64,56*56*128, 28*28*256, 14*14*512, 7*7*512, 25088, 1024]
        # print("The shape of data size is {0} and len pre is {1} workload is: {2}".format(
        #     data_size.shape, len(pre), len(workload)
        # ))
        # for i in range(len(pre)):
        #     Datasize.append(np.max(data_size[i, :]))
        Datasize = [Datasize]
        Datasize = np.array(Datasize)*4/(1024*1024)
        Input_datasize = 224 * 224 * 3 / (1024 * 1024)
        # print(Datasize)
        Workload = [workload]
        N = len(pre)
        Graph_depth = 6
    if graph_type == 2:
        pre = [[], [0], [0], [1], [2], [3], [4], [5,6]]
        data_size = np.loadtxt(os.path.join(local_dir, 'vgg16_boost_vgg19_datasize.txt'))
        data_size[0][0] = 224 * 224 * 3
        data_size = data_size / (1024 * 1024)
        workload = '0.013147159	0.118726106	0.120634575	0.164513698	0.159632201	0.252752676	0.188871536	0.1407831'.split()
        workload = [float(tmp) for tmp in workload]
        workload = [tmp * 4.0 for tmp in workload]

        Datasize = []
        # print("The shape of data size is {0} and len pre is {1} workload is: {2}".format(
        #     data_size.shape, len(pre), len(workload)
        # ))
        for i in range(len(pre)):
            Datasize.append(np.max(data_size[i, :])*4)
        Datasize = [Datasize]
        Input_datasize = 224 * 224 * 3 / (1024 * 1024)
        # print(Datasize)
        Workload = [workload]
        N = len(pre)
        Graph_depth = 5

    if graph_type == 3:
        pre = [[], [0], [0], [1, 2], [3], [3, 4], [5], [5, 6], [7], [7], [8, 9], [10], [10, 11],
               [12], [12, 13], [14], [14, 15], [16], [16], [17, 18], [19], [19, 20], [21], [21, 22],
               [23], [23, 24], [25], [25, 26], [27], [27, 28], [29], [29], [30, 31], [32], [32, 33],
               [34], [34, 35], [36]]
        data_size = np.loadtxt(os.path.join(local_dir, 'resnet50_datasize.txt'))
        data_size[0][0] = 224 * 224 * 3
        data_size = data_size / (1024 * 1024)
        workload = '0.066439695	0.051343827	0.038689017	0.336932201	0.082959342	0.337392921	0.083378024	0.343366885	0.076267123	0.074381781	0.202175264	0.055702724	0.194068999	0.051175013	0.195705881	0.054172029	0.194763641	0.04793674	0.044750166	0.125573778	0.036624289	0.124879093	0.036311903	0.125546317	0.037398782	0.125342145	0.037574887	0.126161351	0.037093277	0.130339127	0.036166987	0.036475015	0.089209728	0.029032798	0.087064009	0.027950749	0.088542018	0.016066232'.split()
        workload = [float(tmp) for tmp in workload]
        workload = [tmp * 4.0 for tmp in workload]

        Datasize = []
        # print("The shape of data size is {0} and len pre is {1} workload is: {2}".format(
        #     data_size.shape, len(pre), len(workload)
        # ))
        for i in range(len(pre)):
            Datasize.append(np.max(data_size[:, i])*4)
        Datasize = [Datasize]
        Input_datasize = 224 * 224 * 3 / (1024 * 1024)
        # print(Datasize)
        Workload = [workload]
        N = len(pre)
        Graph_depth = 25

    if graph_type == 4:
        pre = [[], [0], [0], [0], [0], [1, 2, 3, 4], [5], [5], [5], [5],
               [6, 7, 8, 9], [10], [10], [10], [11, 12, 13], [14], [14],
               [14], [14], [15, 16, 17, 18], [19], [19], [19], [20, 21, 22],
               [23], [23], [23], [24, 25, 26], [27], [27], [27], [28, 29, 30],
               [31]]
        data_size = np.loadtxt(os.path.join(local_dir, 'openface.txt'))
        data_size[0][0] = 224 * 224 * 3
        data_size = data_size / (1024 * 1024)
        workload = '0.024611516	0.008945174	0.007231941	0.00775403	0.007250237	0.021307387	0.010021753	0.009001012	0.008428769	0.008337641	0.025387111	0.01052021	0.009481444	0.008310304	0.014070501	0.007180567	0.006811976	0.006734285	0.006838336	0.014363494	0.007285872	0.00706141	0.005310335	0.007902012	0.004875331	0.00447156	0.004709363	0.006109929	0.004855528	0.00402976	0.004211988	0.006221547	0.002709641'.split()
        workload = [float(tmp) for tmp in workload]
        workload = [tmp * 4.0 for tmp in workload]

        Datasize = []
        # print("The shape of data size is {0} and len pre is {1} workload is: {2}".format(
        #     data_size.shape, len(pre), len(workload)
        # ))
        for i in range(len(pre)):
            Datasize.append(np.max(data_size[:, i])*4)
        Datasize = [Datasize]
        Input_datasize = 224 * 224 * 3 / (1024 * 1024)
        # print(Datasize)
        Workload = [workload]
        N = len(pre)
        Graph_depth = 20

def setBudgetSep(bug):
    global Budget
    Budget = bug

def setSepUploadRate(rate):
    global upload_rate_between_mobile_edge
    upload_rate_between_mobile_edge = rate

def setDatasize(image_count):
    global Datasize
    Datasize = image_count * Datasize

if __name__ == "__main__":
    import numpy as np
    #
    # pre = [[], [0], [0], [1, 2], [3], [3, 4], [5], [5, 6], [7], [7], [8, 9], [10], [10, 11],
    #        [12], [12, 13], [14], [14, 15], [16], [16], [17, 18], [19], [19, 20], [21], [21, 22],
    #        [23], [23, 24], [25], [25, 26], [27], [27, 28], [29], [29], [30, 31], [32], [32, 33],
    #        [34], [34, 35], [36]]
    # data_size = np.loadtxt('resnet50_datasize.txt')
    # data_size[0][0] = 224 * 224 * 3
    # data_size = data_size / (1024 * 1024)
    # workload = '0.066439695	0.051343827	0.038689017	0.336932201	0.082959342	0.337392921	0.083378024	0.343366885	0.076267123	0.074381781	0.202175264	0.055702724	0.194068999	0.051175013	0.195705881	0.054172029	0.194763641	0.04793674	0.044750166	0.125573778	0.036624289	0.124879093	0.036311903	0.125546317	0.037398782	0.125342145	0.037574887	0.126161351	0.037093277	0.130339127	0.036166987	0.036475015	0.089209728	0.029032798	0.087064009	0.027950749	0.088542018	0.016066232'.split()
    # workload = [float(tmp) for tmp in workload]
    # workload = [tmp * 4.0 for tmp in workload]
    #
    # Datasize = []
    # print("The shape of data size is {0} and len pre is {1} workload is: {2}".format(
    #     data_size.shape, len(pre), len(workload)
    # ))
    # for i in range(len(pre)):
    #     Datasize.append(np.max(data_size[i, :]))
    # Datasize = [Datasize]
    #
    # Input_datasize = 224*224*3 / (1024*1024)
    #
    # Workload = [workload]
    #
    # N = len(pre)
    #
    # Graph_depth = 20
    # test_getnexttask_list()
    # test_bfs()
    # test_find_maxmun_latency()
    # test_ftpas()
    # test_get_offloading_policy()
    # test_model()

    setBudgetSep(200)
    # setTaskGraphGSep(3)
    setSepUploadRate(10)
    # setDatasize(2)
    tmpre = [[], [0], [1], [1], [2, 3], [3], [4,5], [5,6]]
    w = np.array([[49.10, 1.15, 35.26, 38.35, 34.19, 35.26, 41.10, 30.17]])
    d = np.array([[35.19, 14.81, 37.04, 12.96, 29.63, 25.93, 24.07, 20.37]])



    import  time
    start_time = time.time()
    # print(get_Xf_Xc_hermes())
    data_list = []
    for i in range(1, len(w[0])+1):
        Datasize = d[:, :i]
        Workload = w[:, :i]
        N = i
        # M = 1
        pre = tmpre[:i]

        if i == 4 or i == 6:
            Datasize = d[:, :i+1]
            Workload = w[:, :i+1]
            N = i+1
            Datasize[0][i] = 0
            Workload[0][i] = 0
            pre = tmpre[:i]
            pre.append([i-2, i-1])

        seq_ue, seq_uc, hemers_ue, hemers_uc, bf_ue, bf_uc, seq_time, hemers_time, bf_time = get_utility()
        data_list.append([seq_ue, seq_uc, hemers_ue, hemers_uc, bf_ue, bf_uc, seq_time, hemers_time, bf_time])
    print("Use time ", time.time()-start_time)

    import pandas as pd

    df = pd.DataFrame(data=data_list)
    df.to_csv("补充实验.csv")




