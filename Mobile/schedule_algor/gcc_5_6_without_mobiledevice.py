import random
import numpy as np
import math

edge_server_num = 4
local_computing_capacity = 1  # 本地计算能力
edge_computing_capacity = 3.6   # 边缘服务器计算能力
cloud_computing_capacity = 10  # 远程云服务器计算能力
pre = [[i - 1] for i in range(6)]
workload = []
sub_task_num = 0
data_size = None
Ptx = 2
gain = 10
noise = 1
eta = 2
wireless_bandwidth = 5 * math.pow(10, 6)  # 5MHz
transmission_rate_remote_cloud = 10 * math.pow(10, 6)  # 与远程云和edge的传输速率10Mbps
R_mnp = None
energy_k = math.pow(10, -11)
alpha_c = 0.6
beta_c = 0.6
alpha_e = 0.5
beta_e = 0.4
price_remote = 0.04
price_edge = [0.03, 0.02, 0.015, 0.01]
P0 = 0.1
max_workload = 100  # 100秒
l_st = 0.1
edge_server_state = [0, 0, 0, 0]
budget_ratio = 0.1

edge_server_workload = None
utility_edge_server = None
Xf = None
XIst = None
Xc = None
budget = 0
TRl_mn = None
TRf_mn = None
TRc_mn = None

# 计算Rmnp
def computer_R_mnp():
    R_mnp = wireless_bandwidth * math.log2(1 + Ptx*gain)
    return R_mnp

# 本地执行时间
def computer_Tl_mn(W_mn):
    return W_mn / local_computing_capacity

# 边缘服务器执行时间
# def computer_Tl_mn(W_mn):
#     return W_mn / edge_computing_capacity

def computer_El_mn(W_mn):
    return energy_k * W_mn * local_computing_capacity * local_computing_capacity

def computer_TFt_mn(Tt_mn, TFl_mn, pre):
    tmp = [0]
    flag = False
    for i in pre:
        if Xc[i] == 0 and np.sum(XIst[:, i]) == 0:
            tmp.append(TFl_mn[i])
            flag = True
    if flag:
        return Tt_mn + max(tmp)
    else:
        return 0

def computer_TFr_mn(Tr_mn, TFf_mn, pre):
    tmp = [0]
    flag = False
    for i in pre:
        if np.sum(XIst[:, i]) != 0:
            tmp.append(TFf_mn[np.where(1==XIst[:, i])[0][0], i])
            flag = True
    if flag:
        return Tr_mn + max(tmp)
    else:
        return 0

def computer_Tr_mn(d_mn):
    return d_mn / transmission_rate_remote_cloud

def computer_Tt_mn(d_mn):
    return d_mn/R_mnp

def computer_TRl_mn(pre,TFl_mn, TFf_mn, TFc_mn):
    tmp = TFl_mn.tolist()
    for i in pre:
        if Xc[i] != 0:
            tmp.append(TFc_mn[i])
        elif np.sum(XIst[:, i]) != 0:
            tmp.append(np.max(TFf_mn[:, i]))
        else:
            tmp.append(TFl_mn[i])
    return max(tmp)

def computer_TRf_mn(TFt_mn, TFf_mn, TFc_mn, pre):
    tmp = np.max(TFf_mn, axis=1).reshape(edge_server_num, 1)
    for i in pre:
        if Xc[i] != 0:
            result = tmp < TFc_mn[i]
            index = 0
            for r in result:
                if r[0]:
                    tmp[index] = TFc_mn[i]
                index += 1
    result = tmp < TFt_mn
    index = 0
    for r in result:
        if r[0]:
            tmp[index] = TFc_mn[i]
        index += 1
    return tmp.reshape(tmp.shape[0])

def computer_TRc_mn(TFt_mn, Tr_mn, TFc_mn, TFr_mn, pre):
    tmp = []
    for i in pre:
        if Xc[i] != 0:
            tmp.append(TFc_mn[i])
        elif np.sum(XIst[:, i]) != 0:
            tmp.append(TFr_mn)
        else:
            tmp.append(TFt_mn + Tr_mn[i])
    return max(tmp)

def computer_TFl_mn(Tl_mn, TRl_mn):
    return Tl_mn + TRl_mn, TRl_mn

def computer_TFf_mn(Tf_mn, TRf_mn):
    return Tf_mn + TRf_mn, TRf_mn

def computer_TFc_mn(Tc_mn, TRc_mn):
    return Tc_mn + TRc_mn, TRc_mn

def computer_Tc_mn(W_mn, fc_q):
    return W_mn/fc_q

def computer_Ec_mn(Tc_mn):
    return (alpha_c * ((cloud_computing_capacity/math.pow(10, 9))**2.5) + beta_c) * Tc_mn

def computer_Tf_mn(W_mn,fe_p):
    return W_mn/fe_p

def computer_Ef_mn(tf_n):
    return (alpha_e * ((edge_computing_capacity/math.pow(10, 9))**2.5) + beta_e) * tf_n

def computer_Tt_mn(d_mn):
    return d_mn/R_mnp

def computer_Es_mn(Tr_mn):
    return P0*Tr_mn

def computer_Et_mn(Tt_mn):
    return  Ptx * Tt_mn

def computer_EI_mn(d_mn):
    return (d_mn/transmission_rate_remote_cloud + l_st) * P0

def computer_C_mn(El_mn, Et_mn, EI_mn, Es_mn):
    tmp = 0.0
    for n in range(sub_task_num):
        exec_in_edge_server_arr = np.where(XIst[:, n]==1)[0]
        if len(exec_in_edge_server_arr) != 0:
            exec_edge_server_index = np.where(XIst[:, n]==1)[0][0]
            tmp = tmp + price_edge[exec_edge_server_index] * np.sum(data_size[:, n]) + Et_mn[n] + EI_mn[n] * np.sum(XIst[1:, n])
        else:
            tmp = tmp + (1 - Xc[n]) * El_mn[n] + (price_remote * np.sum(data_size[:, n]) + Et_mn[n] + Es_mn[n]) * Xc[n]
    return tmp

def computer_Ue(Es_mn, Ee_mn, Ei_mn):
    tmp = []
    for sub_task_index in range(sub_task_num):
        exec_in_edge_server_arr = np.where(XIst[:, sub_task_index]==1)[0]
        if len(exec_in_edge_server_arr) != 0:
            tmp.append(price_edge[exec_in_edge_server_arr[0]] * np.sum(data_size[:, sub_task_index]) * np.sum(XIst[:, sub_task_index])
            - Es_mn[sub_task_index] * Xc[sub_task_index] - Ee_mn[sub_task_index] * np.sum(XIst[:, sub_task_index])
            - Ei_mn[sub_task_index] * np.sum(XIst[1:, sub_task_index]))
    if len(tmp) == 0:
        return 0
    return min(tmp)
def get_min_workload_index(Ec_mn):
    tmp = np.where(Ec_mn * Xc>0)[0]
    min_index = 0
    min_value = 10000
    for index in tmp:
        if min_value > (Ec_mn * Xc)[index]:
            min_index = index
    return min_index

def get_data(list):
    if type(list) == type([]):
        return list[0]
    return list

def init_param(algorithm, task, pre_local, data_size_local, workload_local, gain_local=gain,
               budget_ratio_local=budget_ratio,
               edge_server_state_local=edge_server_state,
               alpha_c_local=alpha_c,
               alpha_e_local=alpha_e,
               beta_e_local=beta_e,
               beta_c_local=beta_c,
               energy_k_local=energy_k,
               price_remote_local=price_remote,
               price_edge_local=price_edge,
               P0_local=P0,
               max_workload_local=max_workload,
               l_st_local=l_st,
               wireless_bandwidth_local=wireless_bandwidth,
               transmission_rate_remote_cloud_local=transmission_rate_remote_cloud,
               eta_local=eta,
               noise_local=noise,
               Ptx_local=Ptx,
               local_computing_capacity_local=local_computing_capacity,
               edge_computing_capacity_local=edge_computing_capacity,
               cloud_computing_capacity_local=cloud_computing_capacity
               ):
    global gain, workload, R_mnp, budget, sub_task_num,\
        edge_server_workload, utility_edge_server,\
        Xf, XIst, Xc, TRc_mn, TRf_mn, TRl_mn,\
        data_size, pre, edge_server_state, budget_ratio, \
        alpha_c, alpha_e, beta_e, beta_c, energy_k,\
        price_remote, price_edge, P0,\
        max_workload, l_st, wireless_bandwidth, transmission_rate_remote_cloud,\
        eta, noise, gain, Ptx,\
        local_computing_capacity, edge_computing_capacity, cloud_computing_capacity
    gain = float(get_data(gain_local))
    if type(get_data(edge_server_state_local)) == type(''):
        edge_server_state = get_data(edge_server_state_local).split(' ')
    else:
        edge_server_state = edge_server_state_local
        edge_server_state = [float(f) for f in edge_server_state]
    budget_ratio = float(get_data(budget_ratio_local))
    alpha_c = float(get_data(alpha_c_local))
    alpha_e = float(get_data(alpha_e_local))
    beta_e = float(get_data(beta_e_local))
    beta_c = float(get_data(beta_c_local))
    energy_k = float(get_data(energy_k_local))
    price_remote = float(get_data(price_remote_local))
    if type(get_data(price_edge_local)) == type(''):
        price_edge = get_data(price_edge_local).split(' ')
        price_edge = [float(f) for f in price_edge]
    else:
        price_edge = price_edge_local
    P0 = float(get_data(P0_local))
    max_workload = float(get_data(max_workload_local))
    l_st = float(get_data(l_st_local))
    wireless_bandwidth = float(get_data(wireless_bandwidth_local))
    transmission_rate_remote_cloud = float(get_data(transmission_rate_remote_cloud_local))
    eta = float(get_data(eta_local))
    noise = float(get_data(noise_local))
    Ptx = float(get_data(Ptx_local))
    local_computing_capacity = float(get_data(local_computing_capacity_local))
    edge_computing_capacity = float(get_data(edge_computing_capacity_local))
    cloud_computing_capacity = float(get_data(cloud_computing_capacity_local))
    pre = pre_local
    data_size = data_size_local
    workload = workload_local
    sub_task_num = len(workload)
    TRl_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    TRf_mn = np.zeros(shape=(edge_server_num, sub_task_num), dtype=np.float32)  # size N
    TRc_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    R_mnp = computer_R_mnp()
    budget = np.sum(data_size) * budget_ratio
    edge_server_workload = np.zeros(edge_server_num)
    utility_edge_server = np.zeros(edge_server_num)
    Xf = np.zeros(shape=sub_task_num)
    XIst = np.zeros((edge_server_num, sub_task_num))
    Xc = np.zeros(shape=sub_task_num)

def run():

    Tl_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    El_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    Tf_mn = np.zeros(shape=sub_task_num, dtype=np.float32) # size N
    Ef_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    Tc_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    Ec_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    Es_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    Tt_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    Tr_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    Et_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    TRl_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    TRf_mn = np.zeros(shape=(edge_server_num, sub_task_num), dtype=np.float32)  # size N
    TRc_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    TFl_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    TFc_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    TFr_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    TFt_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    TFf_mn = np.zeros(shape=(edge_server_num, sub_task_num), dtype=np.float32)  # size N
    EI_mn = np.zeros(shape=sub_task_num, dtype=np.float32)  # size N
    C_mn = 0
    TFf_mn[:, 0] = edge_server_state

    for sub_task_index in range(sub_task_num):
        sub_task_pre = pre[sub_task_index]

        # 分别计算任务在本地执行、边缘设备执行、云端执行所需要的时间和能耗
        Tl_mn[sub_task_index] = computer_Tl_mn(workload[sub_task_index])
        El_mn[sub_task_index] = computer_El_mn(workload[sub_task_index])
        Tc_mn[sub_task_index] = computer_Tc_mn(workload[sub_task_index], cloud_computing_capacity)
        Ec_mn[sub_task_index] = computer_Ec_mn(Tc_mn[sub_task_index])
        Tf_mn[sub_task_index] = computer_Tf_mn(workload[sub_task_index], edge_computing_capacity)
        Ef_mn[sub_task_index] = computer_Ef_mn(Tf_mn[sub_task_index])

        if sub_task_pre[0] == -1: # 首任务默认本地执行
            TRl_mn[sub_task_index] = 0.0
            TRc_mn[sub_task_index] = 0.0
            continue
        else:  # 根据论文公式计算TRl_mn，TRf_mn，TRc_mn以及TFl_mn，TFf_mn，TFc_mn
            Tt_mn[sub_task_index] = computer_Tt_mn(sum(data_size[:, sub_task_index]))
            Tr_mn[sub_task_index] = computer_Tr_mn(sum(data_size[:, sub_task_index]))
            TFt_mn[sub_task_index] = computer_TFt_mn(Tt_mn[sub_task_index], TFl_mn, pre[sub_task_index])
            TFr_mn[sub_task_index] = computer_TFr_mn(Tr_mn[sub_task_index], TFf_mn, pre[sub_task_index])
            TRl_mn[sub_task_index] = computer_TRl_mn(pre[sub_task_index], TFl_mn, TFf_mn, TFc_mn)
            TRf_mn[:, sub_task_index] = computer_TRf_mn(TFt_mn[sub_task_index], TFf_mn, TFc_mn, pre[sub_task_index])
            TRc_mn[sub_task_index] = computer_TRc_mn(TFt_mn[sub_task_index], Tr_mn, TFc_mn, TFr_mn[sub_task_index], pre[sub_task_index])
        TFl_mn[sub_task_index], lastTFl_mn = computer_TFl_mn(Tl_mn[sub_task_index], TRl_mn[sub_task_index])
        TFf_mn[:, sub_task_index], lastTFf_mn = computer_TFf_mn(Tf_mn[sub_task_index], TRf_mn[:, sub_task_index])
        TFc_mn[sub_task_index], lastTFc_mn = computer_TFc_mn(Tc_mn[sub_task_index], TRc_mn[sub_task_index])


        if price_remote * sum(data_size[:, sub_task_index]) / Ec_mn[sub_task_index] >= 1 \
                and TFc_mn[sub_task_index] < np.min(TFf_mn[:, sub_task_index]):
            # 子任务在云端执行
            XIst[:, sub_task_index] = 0
            Xc[sub_task_index] = 1
            TFl_mn[sub_task_index] = lastTFl_mn
            TFf_mn[:, sub_task_index] = lastTFf_mn
            Tt_mn[sub_task_index] = computer_Tt_mn(sum(data_size[:, sub_task_index]))
            Tr_mn[sub_task_index] = computer_Tr_mn(sum(data_size[:, sub_task_index]))
            Es_mn[sub_task_index] = computer_Es_mn(Tr_mn[sub_task_index])
            Et_mn[sub_task_index] = computer_Et_mn(Tt_mn[sub_task_index])
        else:
            # 子任务在边缘设备执行
            XIst[0, sub_task_index] = 1
            Xc[sub_task_index] = 0
            TFc_mn[sub_task_index] = lastTFc_mn
            TFf_mn[:, sub_task_index] = lastTFf_mn
            TFf_mn[0, sub_task_index] += Tf_mn[sub_task_index]
            Tt_mn[sub_task_index] = computer_Tt_mn(sum(data_size[:, sub_task_index]))
            Et_mn[sub_task_index] = computer_Et_mn(Tt_mn[sub_task_index])

            # 选择负载最小的边缘设备
            min_edge_server_index = np.where(edge_server_workload==np.min(edge_server_workload))[0][0]
            if edge_server_workload[min_edge_server_index] + workload[sub_task_index] < max_workload:
                edge_server_workload[min_edge_server_index] += workload[sub_task_index]
                if min_edge_server_index != 0:
                    XIst[0, sub_task_index] = 0
                    XIst[min_edge_server_index, sub_task_index] = 1
                    TFf_mn[:, sub_task_index] = lastTFf_mn
                    TFf_mn[min_edge_server_index, sub_task_index] += Tf_mn[sub_task_index]
                    EI_mn[sub_task_index] = computer_EI_mn(np.sum(data_size[:, sub_task_index]))
        # 根据论文计算执行任务成本
        C_mn = computer_C_mn(El_mn, Et_mn, EI_mn, Es_mn)
    # 进行迭代，将超出预算在云端执行的任务置为边缘设备执行，iter避免死循环
    iter = 100
    while C_mn > budget and iter > 0:
        iter -= 1
        min_edge_server_index = np.where(edge_server_workload==np.min(edge_server_workload))[0][0]
        min_workload_index = get_min_workload_index(Ec_mn)
        # 不满足约束12
        if edge_server_workload[min_edge_server_index] + workload[min_workload_index] < max_workload:
            break
        if np.sum(Xc) != 0:
            Xc[min_workload_index] = 0
            XIst[min_edge_server_index, min_workload_index] = 1
            EI_mn[min_workload_index] = computer_EI_mn(np.sum(data_size[:, min_workload_index]))
        if np.sum(Xc) == 0:
            min_price_edge_index = np.where(price_edge==min(price_edge))[0][0]
            max_price_edge_index = np.where(price_edge==max(price_edge))[0][0]
            target_task_index = np.where(1==XIst[max_price_edge_index, :])[0][0]
            XIst[min_price_edge_index, target_task_index] = 1
            XIst[max_price_edge_index, target_task_index] = 0

    # 计算边缘设备的效用，尽量让边缘设备满足效用大于零
    iter = 100
    while computer_Ue(Es_mn, Ef_mn, EI_mn) < 0 and iter > 0:
        iter -= 1
        Es_mn_largethan_Ef_mn_array = np.where(Es_mn/Ef_mn>1)[0]
        if len(Es_mn_largethan_Ef_mn_array) == 0:
            break
        target_task_index = Es_mn_largethan_Ef_mn_array[0]
        min_edge_server_index = np.where(edge_server_workload==np.min(edge_server_workload))[0][0]
        if np.sum(Xc) != 0 and np.max(Es_mn / Ef_mn) > 1 and edge_server_workload[min_edge_server_index] + workload[target_task_index] < max_workload:
            Xc[target_task_index] = 0
            XIst[min_edge_server_index, target_task_index] = 1

    print("XIst", XIst)
    print("Xc", Xc)
    return Xc, XIst

if __name__ == '__main__':
    pre = [[-1], [0], [0], [1, 2], [3], [3, 4], [5], [5, 6], [7], [7], [8, 9], [10], [10, 11],
           [12], [12, 13], [14], [14, 15], [16], [16], [17, 18], [19], [19, 20], [21], [21, 22],
           [23], [23, 24], [25], [25, 26], [27], [27, 28], [29], [29], [30, 31], [32], [32, 33],
           [34], [34, 35], [36]]
    data_size = np.loadtxt('openface.txt')
    data_size[0][0] = 224 * 224 * 3
    data_size = data_size / (1024*1024)
    workload = '0.024611516    0.008945174    0.007231941    0.00775403 0.007250237    0.021307387    0.010021753    0.009001012    0.008428769    0.008337641    0.025387111    0.01052021 0.009481444    0.008310304    0.014070501    0.007180567    0.006811976    0.006734285    0.006838336    0.014363494    0.007285872    0.00706141 0.005310335    0.007902012    0.004875331    0.00447156 0.004709363    0.006109929    0.004855528    0.00402976 0.004211988    0.006221547    0.002709641'.split()
    workload = [float(tmp) for tmp in workload]
    workload = [tmp*4.0 for tmp in workload]
    init_param('gcc', 'vgg', pre, data_size, workload)
    run()