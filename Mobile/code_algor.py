'''
作者: longxin
时间: 2015-06-01
版本号: 1.0
修改者：
描述：
'''
import numpy as np
import math
import pandas as pd
import random

W1 = 1
W2 = 1
W3 = 1

h = [0.0002177070196967228, 0.00012187730763878534, 0.0004890301600444685, 0.00018420554934396363, 0.0003619520665343829, 0.0003733842521483039, 0.00035446993230631515]
# h = [(np.random.normal(0.8, 0.2, 1)[0]) ** 2 * (50 ** -2) for i in range(0, 7)]
# randomindex = random.randint(0, 6)
randomindex = 3
# h = [(random.random()) ** 2 * (50 ** -2) for i in range(0, 7)]
def computer_Rmnp(n, download):
    import random
    '''
    计算上传的速率
    :param W:  无线带宽，一般为10m
    :param Ptx_mnp: 功耗
    :param h_mnp:  信道增益
    :param you_mp:  参数， 取0.1
    :param Ptx:  功耗数组，代表这个设备的功耗一般都是一样的功耗
    :param h: 信道增益，可能对于不同人物信道增益是不同的
    :return: 上传的速率
    '''
    Sumph = 0.0
    if download:
        W = 1
    else:
        W = 10
    Ptx_mnp = 0.1


    h_mnp = h[randomindex]
    you_mp = 0.008

    Ptx = [0.1 for i in range(0, n)]
    for i in range(0, len(Ptx)):
        for j in range(0, len(h)):
            if Ptx[i] != Ptx_mnp and h[j] != h_mnp:
                Sumph = Sumph + Ptx[i] * h[j]

    # print("the sudu from mobile to egde is:\t", W * math.log2(1.0 + (Ptx_mnp * h_mnp) / (you_mp**2 + Sumph)))
    return W * math.log2(1.0 + (Ptx_mnp * h_mnp) / (you_mp**2 + Sumph))


def computer_tl_mn(Wmn):
    '''
    计算在本地的执行时间
    :param Wmn: 设备m上任务n的计算量的大小
    :param fl_m: 设备m的计算频率
    :return:
    '''
    fl_m = 0.1

    return Wmn / fl_m

def computer_El_mn(Wmn):
    '''
    计算任务在本地执行需要消耗的能量
    :param Wmn:  设备m上任务n的计算量的大小
    :param fl_m: 设备m计算的频率
    :return:
    '''
    # 参数 k
    k = 0.05
    fl_m = 0.1
    # print("本地任务上计算消耗的能量:\t",  k * Wmn * fl_m * fl_m)
    return k * Wmn * fl_m * fl_m

def computer_tf_mn(Wmn):
    '''
    计算在fog上计算任务需要的时间
    :param Wmn: 设备m上任务n的计算量的大小
    :param ff_p:  fog服务器计算的频率
    :return:
    '''
    ff_p = 1
    return Wmn / ff_p

def computer_Ef_mn(tf_mn):
    '''
    计算在fog上计算任务需要消耗的能量
    :param ff_p: fog计算的频率
    :param tf_mn: fog上计算任务的时间
    :return:
    '''
    # 参数
    alpha_f = 0.01
    beta_f = 0.001
    eci = 0.1
    ff_p = 1.024
    # print("fog上计算消耗的能量:\t", (alpha_f*(ff_p**eci) + beta_f) * tf_mn)
    return (alpha_f*(ff_p**eci) + beta_f) * tf_mn


def computer_Ec_mn(tc_mn):
    '''
    计算在远程云上执行任务所需要消耗的能量
    :param fc_q:  远程云的计算频率
    :param tc_mn:  远程云上的计算时间
    :return:
    '''
    # 参数
    alpha_c = 0.01
    beta_c = 0.001
    eci = 0.1
    fc_q = 4.096
    # print("在远程云上计算消耗的能量:\t", (alpha_c*(fc_q**eci) + beta_c) * tc_mn)
    return (alpha_c*(fc_q**eci) + beta_c) * tc_mn

def computer_tc_mn(Wmn):
    '''
    计算设备m上的任务n在远程云上的执行时间
    :param Wmn:  设备m上任务n的计算量的大小
    :param fc_q:  远程云的计算频率
    :return:
    '''
    fc_q = 4
    return Wmn / fc_q

def computer_tt_mn(dmn, Rmn):
    '''
    计算设备m上任务n从移动设备迁移到Edgeserver的时间
    :param dmn:  设备m上任务n的数据量的大小
    :param Rmn:  设备m将任务n迁移到Edgeserver的时间
    :return:
    '''
    return dmn / Rmn

def computer_Et_mn(tt_mn):
    '''
    计算设备m上任务n从移动设备迁移到Edgeserver消耗的能量
    :param Ptx_mn:  设备m迁移任务n的功率
    :param tt_mn:  设备m迁移任务n花费的时间
    :return:
    '''
    Ptx_mn = 0.1
    # print("传输任务到fog上需要消耗的能量:\t",Ptx_mn * tt_mn )
    return Ptx_mn * tt_mn

def computer_tt2_mn(dmn, R2mn):
    '''
    计算设备m上任务n从EdgeServer 传到手机的需要的时间
    :param dmn: 设备m上的任务n数据量的大小
    :param R2mn: 从EdgerServer传输数据到moble的传输速度
    :return:
    '''
    return dmn / R2mn

def computer_Et2_mn(tt2_mn):
    '''
    计算设备m上的任务n将任务从EdgerServer传输到mobile需要消耗的能量
    :param tt2_mn: 设备m上的任务n从EdgeServer传输到mobile需要消耗的时间
    :return:
    '''
    Ptx_mn = 0.1
    return Ptx_mn * tt2_mn

def computer_tr_mn(dmn):
    '''
    计算设备m上的任务n将任务从Edgeserver传到Remote cloud上所需要的时间
    :param dmn:  设备m上的任务n上的数据量的大小
    :return:
    '''
    # 从EdgeServer传到remotecloud的带宽
    w = 51200
    return dmn / w

def computer_Er_mn(tr_mn):
    '''
    计算设备m上的任务n将任务从Edgeserver传到remotecloud上所需要消耗的能量
    :param tr_mn:  设备m上的任务n将任务从EdgeServer传到Remotecloud所需要的时间
    :return:
    '''
    P0 = 1
    # print("传输任务到remotecloud需要消耗的能量:\t", P0 * tr_mn)
    return P0 * tr_mn


def computer_tr2_mn(dmn):
    '''
    计算设备m上的任务n从remote cloud迁移到EdgeServer的需要的时间
    :param dmn:  设备m上任务n上的数据量的大小
    :return:
    '''
    w = 512000
    return dmn / w

def computer_Er2_mn(tr2_mn):
    '''
    计算设备m上任务n从remote cloud迁移到EdgeServer所需要消耗的能量
    :param tr2_mn: 设备m上的任务n从remote cloud 迁移到EdgeServer所需要的时间
    :return:
    '''
    P0 = 1
    return P0 * tr2_mn

def computer_tfl_mn(Xm, Xf, Xc, d, w, pre, m, n):
    '''
    假设设备m上的任务n在本地执行，它所需要的时间
    :param Xm: 本地的任务防止策略
    :param Xf: fog上的任务放置策略
    :param Xc: 云上的任务放置策略
    :param d:  所有任务的数据量的大小
    :param w:  所有任务的计算量的大小
    :param pre: 任务n的前面的任务列表
    :param m: the device number
    :param n: the task number
    :return:
    '''
    # 首先计算pre中的传输时间
    pret = []
    for k in pre:
        Rmn = computer_Rmnp(len(d[m-1]), True)
        tt2_mk = computer_tt2_mn(d[m-1][n], Rmn)
        tr2_mk = computer_tr2_mn(d[m-1][n])

        pret.append(Xf[m-1][k]*tt2_mk + Xc[m-1][k]*(tt2_mk + tr2_mk))

    # 得到最大的值
    maxPret = max(pret)
    maxK = pret.index(maxPret)

    # 计算本地计算需要的时间
    tl_mn = computer_tl_mn(w[m-1][n])

    # 返回结果
    return pret[maxK] + tl_mn


def computer_tff_mn(Xm, Xf, Xc, d, w, pre, m, n):
    '''
    假设设备m上的任务n在fog上执行，所需要的的时间
    :param Xm:  本地的任务执行策略
    :param Xf:  fog上的任务执行策略
    :param Xc:  remote cloud上的任务执行策略
    :param d:  所有任务的数据量的大小
    :param w:  所有任务的计算量的大小
    :param pre:  任务的前置数组
    :param m:  mobile device number
    :param n:  task number
    :return:
    '''
    pret = []
    for k in pre:
        Rmn = computer_Rmnp(len(d[m-1]), False)
        tt_mn = computer_tt_mn(d[m-1][n], Rmn)
        tr_mn = computer_tr2_mn(d[m-1][n])
        if len(pre) == 1 and k == -1:
            pret.append(1*tt_mn)
        else:
            pret.append(Xm[m-1][k]*tt_mn + Xc[m-1][k]*tr_mn)

    # get the max value
    maxPret = max(pret)
    maxK = pret.index(maxPret)

    # computer the fog excution time
    tf_mn = computer_tf_mn(w[m-1][n])

    # return the result
    return tf_mn + pret[maxK]

def computer_tfc_mn(Xm, Xf, Xc, d, w, pre, m, n):
    '''
    假设设备m上的任务n在云端上执行任务需要的时间
    :param Xm:  本地的5任务执行策略
    :param Xf:  fog上的任务执行策略
    :param Xc:  云端上的任务执行策略
    :param d:  任务数据量大小的数组
    :param w:  任务计算量大小的数组
    :param pre:  任务的dag前置数组
    :param m:  device number
    :param n:  task number
    :return:
    '''
    # 计算在pre中的值
    pret = []
    for k in pre:
        Rmn = computer_Rmnp(len(d[m-1]), False)
        tt_mn = computer_tt_mn(d[m-1][n], Rmn)
        tr_mn = computer_tr_mn(d[m-1][n])

        if len(pre) == 1 and k == -1:
            pret.append(tt_mn+tr_mn)
        else:
            pret.append(Xm[m-1][k]*(tt_mn + tr_mn) + Xf[m-1][k]*tr_mn)
    #  get the max value
    maxPret = max(pret)
    maxK = pret.index(maxPret)

    # computer the cloud excution time
    tc_mn = computer_tc_mn(w[m-1][n])
    # return the result
    return pret[maxK] + tc_mn

def computer_efl_mn(Xm, Xf, Xc, d, w, pre, m, n):
    '''
    假设设备m上的任务n在本地执行，所需要的消耗的能量
    :param Xm:  本地任务执行策略
    :param Xf:  fog上任务的执行策略
    :param Xc:  remote cloud上的任务执行策略
    :param d:  任务数据量的大小数组
    :param w:  任务计算量大小的数组
    :param pre:  设备m上的任务n上的前置r任务编号数组
    :param m:  device number
    :param n:  task number
    :return:
    '''
    prete = 0.0
    # 对前置任务传输消耗能量求和
    for k in pre:
        Rmn = computer_Rmnp(len(d[m-1]), True)
        tt2_mn = computer_tt2_mn(d[m-1][n], Rmn)
        et2_mn = computer_Et2_mn(tt2_mn)

        tr2_mn = computer_tr2_mn(d[m-1][n])
        er2_mn = computer_Er2_mn(tr2_mn)

        prete = prete + W2*Xf[m-1][k]*et2_mn + W3*Xc[m-1][n]*(et2_mn + er2_mn)

    em_mn = computer_El_mn(w[m-1][n])
    # 返回结果
    return em_mn + prete

def computer_eff_mn(Xm, Xf, Xc, d, w, pre, m, n):
    '''
    假设任务在fog执行，则需要消耗的能量
    :param Xm: 任务在本地的执行策略
    :param Xf: 任务在fog节点上的执行策略
    :param Xc: 任务在remote cloud上的执行策略
    :param d: 任务数据量大小的数组
    :param w: 任务计算量大小的数组
    :param pre: 任务的前置数组
    :param m: device number
    :param n: task number
    :return:
    '''
    prete = 0.0
    for k in pre:
        Rmn = computer_Rmnp(len(d[m-1]), False)
        tt_mn = computer_tt_mn(d[m-1][n], Rmn)
        et_mn = computer_Et_mn(tt_mn)

        tr_mn = computer_tr2_mn(d[m-1][n])
        er_mn = computer_Er2_mn(tr_mn)

        if len(pre) == 1 and k == -1:
            prete = prete + W1 * 1 * et_mn + W3 * Xc[m - 1][n] * er_mn
        else:
            prete = prete + W1*Xm[m-1][k]*et_mn + W3*Xc[m-1][n]*er_mn

    # 计算任务在本地的执行消耗
    tf_mn = computer_tf_mn(w[m-1][n])
    ef_mn = computer_Ef_mn(tf_mn)
    # return the result
    return prete + ef_mn

def computer_efc_mn(Xm, Xf, Xc, d, w, pre, m, n):
    '''
    计算在remote cloud上执行任务的消耗
    :param Xm:  任务在本地的执行策略
    :param Xf:  任务在fog上的执行策略
    :param Xc:  任务在remote cloud上的执行策略
    :param d:  任务数据量的大小
    :param w:  任务计算量的大小
    :param pre:  前置数组
    :param m:  device number m
    :param n:  task number n
    :return:
    '''
    # 计算前置任务总的能量消耗
    prete = 0.0
    for k in pre:
        Rmn = computer_Rmnp(len(d[m-1]), False)
        tt_mn = computer_tt_mn(d[m-1][n], Rmn)
        et_mn = computer_Et_mn(tt_mn)

        tr_mn = computer_tr_mn(d[m-1][n])
        er_mn = computer_Er_mn(tr_mn)

        if len(pre) == 1 and k == -1:
            prete = prete + W1 * 1 * (et_mn + er_mn) + W2 * Xf[m - 1][k] * er_mn
        else:
            prete = prete + W1*Xm[m-1][k]*(et_mn + er_mn) + W2*Xf[m-1][k]*er_mn


    # 计算在本地的执行消耗
    tc_mn = computer_tc_mn(w[m-1][n])
    ec_mn = computer_Ec_mn(tc_mn)
    # return the result
    return prete + ec_mn


def computer_Ufm(w, Xf, Xc, d):
    '''
    计算Ederserver的untility
    :param w:  任务的计算量大小的数组
    :param Xf:  在fog上的执行策略
    :param Xc:  在remote cloud上的执行策略
    :return:
    '''
    # 上传所消耗的能量
    tmpXf = np.array(Xf)
    Er = np.zeros(shape=tmpXf.shape)

    # 计算上传所需要消耗的能量
    for i in range(0, tmpXf.shape[1]):
        tr_mn = computer_tr_mn(d[tmpXf.shape[0]-1][i])
        Er[tmpXf.shape[0]-1][i] = computer_Er_mn(tr_mn)

    # 计算utinity
    ufm = 0.0
    # 获得价格
    pcq = 0.03
    for i in range(0, tmpXf.shape[1]):
        ufm = ufm + pcq * w[tmpXf.shape[0]-1][i] * Xf[tmpXf.shape[0]-1][i]

    for i in range(0, tmpXf.shape[1]):
        ufm = ufm - Er[tmpXf.shape[0]-1][i]*Xc[tmpXf.shape[0]-1][i]

    return ufm



def computer_AllE(d, w, pre, m, n, tasks, Xc, Xf, Xm):
    '''
    重新计算所有的能耗数组
    :param d:  任务数据量大小数组
    :param w:  任务计算量大小数组
    :param pre:  任务前置数组
    :param m:  设备数量
    :param n:  任务数量
    :param tasks:  任务执行的顺序
    :param Xc:  remote cloud执行策略
    :param Xf:  EdgeServer执行策略
    :param Xm:  mobile本地执行策略
    :return:
    '''
    tfl = np.zeros(shape=(m, n))
    tff = np.zeros(shape=(m, n))
    tfc = np.zeros(shape=(m, n))

    efl = np.zeros(shape=(m, n))
    eff = np.zeros(shape=(m, n))
    efc = np.zeros(shape=(m, n))

    # 对于每一个任务选择消耗能量最小
    # 默认m为0
    for task in tasks:
        # 计算在本地上执行需要消耗的能量
        efl[m-1][task] = computer_efl_mn(Xm, Xf, Xc, d, w, pre[task], m, task)
        # 计算在EdgerServer上执行需要消耗的能量
        eff[m-1][task] = computer_eff_mn(Xm, Xf, Xc, d, w, pre[task], m, task)
        # 计算在RemoteCloud上执行需要消耗的能量
        efc[m-1][task] = computer_efc_mn(Xm, Xf, Xc, d, w, pre[task], m, task)


    return efl, eff, efc

def computer_AllT(d, w, pre, m, n, tasks, Xc, Xf, Xm):
    '''
    根据当前的策略重新计算所有的时间
    :param d: 任务数据量的大小
    :param w: 任务计算量的大小
    :param pre: 任务的前置数组
    :param m: 设备的数量
    :param n: 任务的数量
    :param tasks: 任务数组，执行的顺序
    :param Xc: 任务在云端的执行策略
    :param Xf: 任务在fog上的执行策略
    :param Xm: 任在本地的执行策略
    :return:
    '''

    # 初始化各个参数
    tfl = np.zeros(shape=(m, n), dtype=np.float32)
    tff = np.zeros(shape=(m, n), dtype=np.float32)
    tfc = np.zeros(shape=(m, n), dtype=np.float32)

    efl = np.zeros(shape=(m, n))
    eff = np.zeros(shape=(m, n))
    efc = np.zeros(shape=(m, n))

    for task in tasks:
        tfl[m-1][task] = computer_tfl_mn(Xm,Xf,Xc,d,w,pre[task],m,task)
        # 计算在本地任务的执行时间

        tff[m-1][task]= computer_tff_mn(Xm,Xf,Xc,d,w,pre[task],m,task)
        # 计算在fog上任务执行的时间

        tfc[m-1][task] = computer_tfc_mn(Xm, Xf,Xc,d,w, pre[task],m,task)
        # 计算任务在cloud上的执行时间

    return tfl,tff, tfc


def gainMethod(d, w, pre, m, n, tasks, buget):
    '''
    使用贪心发法求解最好的策略
    :param d: 数据量大小的数组  size m * n
    :param w:  任务量大小的数组 size m * n
    :param pre:  任务的前置数组， size m * n * n
    :param m: device number
    :param n: task number
    :param tasks: 任务的执行顺序
    :return:
    '''

    # 初始化各个数组
    print("gain method begin")

    Xc = np.zeros(shape=(m, n))
    Xf = np.zeros(shape=(m, n))
    Xm = np.zeros(shape=(m, n))

    tfl = np.zeros(shape=(m, n))
    tff = np.zeros(shape=(m, n))
    tfc = np.zeros(shape=(m, n))

    efl = np.zeros(shape=(m, n))
    eff = np.zeros(shape=(m, n))
    efc = np.zeros(shape=(m, n))

    # 对于每一个任务选择消耗能量最小
    # 默认m为0
    for task in tasks:

        # 计算在本地上执行需要消耗的能量
        efl[m-1][task] = computer_efl_mn(Xm, Xf, Xc, d, w, pre[task], m, task)
        # 计算在EdgerServer上执行需要消耗的能量
        eff[m-1][task] = computer_eff_mn(Xm, Xf, Xc, d, w, pre[task], m, task)
        # 计算在RemoteCloud上执行需要消耗的能量
        efc[m-1][task] = computer_efc_mn(Xm, Xf, Xc, d, w, pre[task], m, task)
        # 选择消耗能量最小的策略
        minE = min(efl[m-1][task], eff[m-1][task], efc[m-1][task])

        # print("for task:\t", task, "本地执行消耗为:/t", efl[m-1][task], "fog上执行消耗能量为:\t", eff[m-1][task], "remote cloud执行消耗能量为:\t", efc[m-1][task])

        # 计算选择在本地上执行需要的时间
        tfl[m-1][task] = computer_tfl_mn(Xm, Xf, Xc, d, w, pre[task], m, task)

        # 计算在EdgeServer执行需要的时间
        tff[m-1][task] = computer_tff_mn(Xm, Xf, Xc, d, w, pre[task], m, task)

        # 计算在remote cloud上需要的时间
        tfc[m-1][task] = computer_tfc_mn(Xm, Xf, Xc, d, w, pre[task], m, task)

        # 选择消耗能量最小的策略
        if minE == efl[m-1][task]:
            Xf[m-1][task] = 1
            continue

        if minE == eff[m-1][task]:
            Xf[m-1][task] = 1
            continue

        if minE == efc[m-1][task]:
            Xc[m-1][task] = 1
            continue
    # print("Xm:\t", Xm, "Xf:\t", Xf, "Xc:\t", Xc)
    # 进行utility和时间上约束的调整
    upf = computer_Ufm(w, Xf, Xc, d)
    total_time = computer_totaltime(Xm, Xf, Xc, tfl, tff, tfc, m, n)
    # total_time = computer_totalTime(tfl, tff, tfc, Xm, Xf, Xc)
    # while upf < 0 or total_time >buget:
    while total_time > buget:
        # if  upf < 0:
        #     Xm,Xf,Xc = chooseFromCloudEdge(Xm, Xf, Xc, d, w, pre, m, n, tasks)
        #     upf = computer_Ufm(w, Xf, Xc, d)
        #     tmptfl, tmptff, tmptfc = computer_AllT(d, w, pre, m, n, tasks, Xc, Xf, Xm)
        #     total_time = computer_totaltime(Xm, Xf, Xc, tmptfl, tmptff, tmptfc, m, n)
        #     # print("due to the upf, change the policy", upf)

        '根据策略重新计算utinitly和时间'
        # total_time = computer_totalTime(tmptfl,tmptff,tmptfc, Xc,Xf,Xm)

        if total_time > buget:
            '''
            如果总的时间大于buget，对于所有任务任务重新计算，选择增加能耗最少，执行时间减少的任务，
            如果任务是从云端迁移到EdgerServer，直接迁移，因为这样不会引起unitnity，如果不是，则选择排名靠前的策略
            '''
            # print("before the fit, the time is:\t", total_time)
            Xm,Xf,Xc = FitTheTimeBuget(Xm,Xf,Xc,d,w,pre, m, n, tasks)
            tmptfl, tmptff, tmptfc = computer_AllT(d, w, pre, m, n, tasks, Xc, Xf, Xm)
            total_time = computer_totaltime(Xm, Xf, Xc, tmptfl, tmptff, tmptfc, m, n)
            upf = computer_Ufm(w, Xf, Xc, d)
            print("fit the time budget ", total_time)
            # print("change the policy one time", total_time)

        # total_time = computer_totalTime(tmptfl, tmptff, tmptfc, Xc, Xf, Xm)

    print("gain method close")
    return Xm,Xf,Xc

def computer_totale(Xm,  Xf,  Xc, efl, eff, efc, m, n):
    totale = 0.0
    for i in range(0, n):
        totale = totale + Xm[m-1][i]*efl[m-1][i] + Xf[m-1][i]*eff[m-1][i] + Xc[m-1][i]*efc[m-1][i]
    return totale

def FitTheTimeBuget( Xm,  Xf, Xc, d, w, pre, m, n,tasks):
    '''
    满足时间上的约束
    算法的贪婪策略为对所有的任务重新计算，选择增加能耗最少，执行时间减少的任务，而且不会引起upf的减少最好
    :param Xm:  本地的任务执行策略
    :param Xf:  fog上的任务执行策略
    :param Xc:  remote cloud上的任务执行策略
    :param d:  任务数据量的大小
    :param w:  任务计算量的大小
    :param pre:  任务的前置任务list
    :param m:  设备的数量
    :param n:  任务的数量
    :param tasks:  任务的执行顺序
    :return:
    '''

    minE = 100000
    minEtask = -1
    minChangeway = -1

    efl,eff,efc = computer_AllE(d, w, pre, m, n, tasks, Xc, Xf, Xm)
    tfl, tff, tfc = computer_AllT(d, w, pre, m, n, tasks, Xc, Xf, Xm)
    initE = computer_totale(Xm, Xf, Xc, efl, eff, efc, m, n)
    initT = computer_totaltime(Xm, Xf, Xc, tfl, tff, tfc, m, n)
    # print("the init time is:\t", initT)
    # print("the init E is:\t", initE, tfl, tff, tfc)
    # initE = sum(efl[m-1] + eff[m-1] + efc[m-1])

    tmpXm = Xm.copy()
    tmpXf = Xf.copy()
    tmpXc = Xc.copy()
    for task in tasks:
        '计算将任务迁移不同地方的能耗增加程度'
        # 如果任务在本地执行
        if Xm[m-1][task] == 1:
            '计算将任务迁移到fog上的能耗变化'
            tmpXm[m-1][task] = 0
            tmpXf[m-1][task] = 1
            tmpefl, tmpeff, tmpefc = computer_AllE(d, w, pre, m, n, tasks, tmpXc, tmpXf, tmpXm)
            tmptfl, tmptff, tmptfc = computer_AllT(d, w, pre, m, n, tasks, tmpXc, tmpXf, tmpXm)
            tmpE = computer_totale(tmpXm, tmpXf, tmpXc, tmpefl, tmpeff, tmpefc, m, n)
            tmptotaltime = computer_totaltime(tmpXm, tmpXf, tmpXc, tmptfl, tmptff, tmptfc, m, n)
            # tmpE = sum(tmpefl[m-1] + tmpeff[m-1] + tmpefc[m-1][n])
            # print("the tmpE - initE is:\t", tmpE - initE)
            # print("the tmp total time is:\t", tmptotaltime, tmptfl, tmptff, tmptfc)

            if tmptotaltime < initT and minE > tmpE - initE:
                minE = tmpE - initE
                minEtask = task
                minChangeway = 2
                # print("one change")
                # print("the tmp total time is:\t", tmptotaltime)

            tmpXm[m-1][task] = 1
            tmpXf[m-1][task] = 0

        elif Xf[m-1][task] == 1:
            '计算将任务迁移到本地上执行的能耗变化'
            tmpXf[m - 1][task] = 0
            tmpXc[m - 1][task] = 1
            tmpefl, tmpeff, tmpefc = computer_AllE(d, w, pre, m, n, tasks, tmpXc, tmpXf, tmpXm)
            tmptfl, tmptff, tmptfc = computer_AllT(d, w, pre, m, n, tasks, tmpXc, tmpXf, tmpXm)
            tmptotaltime = computer_totaltime(tmpXm, tmpXf, tmpXc, tmptfl, tmptff, tmptfc, m, n)
            tmpE = computer_totale(tmpXm, tmpXf, tmpXc, tmpefl, tmpeff, tmpefc, m, n)
            # print("the tmpE - initE is:\t", tmpE - initE)
            # print("the tmp total time is:\t", tmptotaltime, tmptfl, tmptff, tmptfc)

            if tmptotaltime < initT and minE > tmpE - initE:
                minE = tmpE - initE
                minEtask = task
                minChangeway = 1
                # print("two change")
                # print("the tmp total time is:\t", tmptotaltime)

            tmpXf[m - 1][task] = 1
            tmpXc[m - 1][task] = 0


        else:
            # '计算将任务放在本地上的能耗变化'
            # tmpXm[m - 1][task] = 1
            # tmpXc[m - 1][task] = 0
            # tmpefl, tmpeff, tmpefc = computer_AllE(d, w, pre, m, n, tasks, tmpXc, tmpXf, tmpXm)
            # tmptfl, tmptff, tmptfc = computer_AllT(d, w, pre, m, n, tasks, tmpXc, tmpXf, tmpXm)
            # tmptotaltime = computer_totaltime(tmpXm, tmpXf, tmpXc, tmptfl, tmptff, tmptfc, m, n)
            # tmpE = computer_totale(tmpXm, tmpXf, tmpXc, tmpefl, tmpeff, tmpefc, m, n)
            # # tmpE = sum(tmpefl[m - 1] + tmpeff[m - 1] + tmpefc[m - 1][n])
            # # print("the tmpE - initE is:\t", tmpE - initE)
            # # print("the tmp total time is:\t", tmptotaltime, tmptfl, tmptff, tmptfc)
            #
            # if tmptotaltime < initT and minE > tmpE - initE:
            #     minE = tmpE - initE
            #     minEtask = task
            #     minChangeway = 1
            #     # print("three change")
            #     # print("the tmp total time is:\t", tmptotaltime)
            #
            # tmpXm[m - 1][task] = 0
            # tmpXc[m - 1][task] = 1

            '计算将任务放在fog上的能耗变化'
            tmpXf[m - 1][task] = 1
            tmpXc[m - 1][task] = 0
            tmpefl, tmpeff, tmpefc = computer_AllE(d, w, pre, m, n, tasks, tmpXc, tmpXf, tmpXm)
            tmptfl, tmptff, tmptfc = computer_AllT(d, w, pre, m, n, tasks, tmpXc, tmpXf, tmpXm)
            tmptotaltime = computer_totaltime(tmpXm, tmpXf, tmpXc, tmptfl, tmptff, tmptfc, m, n)
            tmpE = computer_totale(tmpXm, tmpXf, tmpXc, tmpefl, tmpeff, tmpefc, m, n)
            # print("the tmpE - initE is:\t", tmpE - initE)
            # print("the tmp total time is:\t", tmptotaltime, tmptfl, tmptff, tmptfc)
            # tmpE = sum(tmpefl[m - 1] + tmpeff[m - 1] + tmpefc[m - 1][n])

            if tmptotaltime < initT and minE > tmpE - initE:
                minE = tmpE - initE
                minEtask = task
                minChangeway = 2
                # print("fourth change")
                # print("the tmp total time is:\t", tmptotaltime)

            tmpXf[m - 1][task] = 0
            tmpXc[m - 1][task] = 1


    '如果在本地执行'
    if Xm[m-1][minEtask] == 1:
        tmpXm[m-1][minEtask] = 0
        tmpXf[m-1][minEtask] = 1
        # print("change")


    '如果在fog上执行'
    if Xf[m-1][minEtask] == 1:
        tmpXc[m-1][minEtask] = 1
        tmpXf[m-1][minEtask] = 0

    '如果在remote cloud上执行'
    if Xc[m-1][minEtask] == 1:
        if minChangeway == 2:
            tmpXc[m-1][minEtask] = 0
            tmpXf[m-1][minEtask] = 1

    # print("the min task is:\t", minEtask)
    # print("the change Xm,Xf,Xc is\t", tmpXm, tmpXf, tmpXc)

    # tmptfl, tmptff, tmptfc = computer_AllT(d, w, pre, m, n, tasks, tmpXc, tmpXf, tmpXm)
    # afterT = computer_totaltime(tmpXm, tmpXf, tmpXc, tmptfl, tmptff, tmptfc, m, n)
    # # print("the change Xm,Xf,Xc is\t", tmpXm, tmpXf, tmpXc)
    # print("after the precess, the t is\t", afterT)
    # afterT = computer_totaltime(tmpXm, tmpXf, tmpXc, tmptfl, tmptff, tmptfc, m, n)
    # print("after the 2 precess the t is \t", afterT)


    return tmpXm,tmpXf,tmpXc

def chooseFromCloudEdge(Xm, Xf,Xc,d, w, pre, m, n, tasks):

    '计算所有从云端迁移到本地的能耗变化'
    Efchange = []
    Emchange = []

    '选择所有在云端的编号'
    CloudTasks = []
    for task, value in enumerate(Xc[m-1]):
        if value == 1:
            CloudTasks.append(task)

    efl,eff,efc = computer_AllE(d,w,pre,m,n,tasks,Xc,Xf,Xm)
    initE = computer_totale(Xm, Xf, Xc, efl, eff, efc, m, n)
    # initE = sum(efl[m]) + sum(eff[m]) + sum(efc[m])
    for task in CloudTasks:
        tmpXc = Xc.copy()
        tmpXf = Xf.copy()
        tmpXm = Xm.copy()

        tmpXc[m-1][task] = 0
        tmpXf[m-1][task] = 1
        tmpefl,tmpeff,tmpefc = computer_AllE(d,w,pre, m,n,tasks,tmpXc,tmpXf,tmpXm)
        fogE = computer_totale(tmpXm, tmpXf, tmpXc, tmpefl, tmpeff, tmpefc, m, n)
        # fogE = sum(tmpefl[m]) + sum(tmpeff[m]) + sum(tmpefc[m])


        tmpXf[m-1][task] = 0
        tmpXm[m-1][task] = 1
        tmpefl, tmpeff, tmpefc = computer_AllE(d, w, pre, m, n, tasks, tmpXc, tmpXf, tmpXm)
        mobileE = computer_totale(tmpXm, tmpXf, tmpXc, tmpefl, tmpeff, tmpefc, m, n)
        # mobileE = sum(tmpefl[m]) + sum(tmpeff[m]) + sum(tmpefc[m])

        Efchange.append(fogE -initE)
        Emchange.append(mobileE - initE)

    '得到能耗增加最少的任务进行迁移'
    minEfChange = min(Efchange)
    minEmChange = min(Emchange)

    if minEfChange < minEmChange:
        index = Efchange.index(minEfChange)
        task = CloudTasks[index]
        Xf[m-1][task] = 1
        Xc[m-1][task] = 0

    else:
        index = Emchange.index(minEmChange)
        task = CloudTasks[index]
        Xf[m-1][task] = 0
        Xc[m-1][task] = 1

    return Xm,Xf, Xc

def getOneStep(Xm, Xf, Xc, m, n, buget):
    import random
    tmptotaltime = 100000
    tmpupf = 0
    tmpE = -1
    while tmptotaltime > buget or tmpupf < 0:
        tmpXc = Xc.copy()
        tmpXf = Xf.copy()
        tmpXm = Xm.copy()
        '随机得到一个任务'
        tmptask = random.randint(0, n - 1)

        '随机选择任务在本地、fog、cloud执行'
        tmpChange = random.randint(1, 3)

        if tmpChange == 1:
            tmpXm[m - 1][tmptask] = 1
            tmpXf[m - 1][tmptask] = 0
            tmpXc[m - 1][tmptask] = 0
        elif tmpChange == 2:
            tmpXm[m - 1][tmptask] = 0
            tmpXf[m - 1][tmptask] = 1
            tmpXc[m - 1][tmptask] = 0
        else:
            tmpXm[m - 1][tmptask] = 0
            tmpXf[m - 1][tmptask] = 0
            tmpXc[m - 1][tmptask] = 1

        '计算能量消耗、upf、任务执行时间'
        tmpefl, tmpeff, tmpefc = computer_AllE(d, w, pre, m, n, tasks, tmpXc, tmpXf, tmpXm)
        tmpE = sum(tmpefl[m - 1] + tmpeff[m - 1] + tmpefc[m - 1])
        tmptfl, tmptff, tmptfc = computer_AllT(d, w, pre, m, n, tasks, tmpXc, tmpXf, tmpXm)
        # print("tmpefl:\t", tmpefl, "tmpeff:\t", tmpeff, "tmpefc:\t", tmpefc)
        tmptotaltime = sum(tmptfl[m - 1] + tmptff[m - 1] + tmptfc[m - 1])
        tmpupf = computer_Ufm(w, Xf, Xc, d)

def simulMethod(d, w, pre, tasks, m, n, T0, buget):
    import random
    import math
    '''
    使用模拟退火求解最优的策略
    :param d: 任务数据量大小的数组
    :param w: 任务计算量大小的数组
    :param pre: 任务执行的前置数组
    :param T0: 起始温度
    :param buget: 
    :return:
    '''
    '初始化各个参数'
    Xm = np.zeros(shape=(m, n))
    Xf = np.zeros(shape=(m, n))
    Xc = np.zeros(shape=(m, n))
    efl = np.zeros(shape=(m, n))
    eff = np.zeros(shape=(m, n))
    efc = np.zeros(shape=(m, n))
    tfl = np.zeros(shape=(m,n))
    tff = np.zeros(shape=(m, n))
    tfc = np.zeros(shape=(m, n))
    print("模拟退火开始")

    '随机得到一个结果'
    # for task in tasks:
    #     case = random.randint(1,3)
    #     if case == 1:
    #         Xm[m-1][task] = 1
    #         Xf[m-1][task] = 0
    #         Xc[m-1][task] = 0
    #     elif case == 2:
    #         Xm[m-1][task] = 0
    #         Xf[m-1][task] = 1
    #         Xc[m-1][task] = 0
    #     else:
    #         Xm[m-1][task] = 0
    #         Xf[m-1][task] = 0
    #         Xc[m-1][task] = 1
    Xm,Xf,Xc = gainMethod(d,w,pre,m,n,tasks,buget)
    # 现在的温度
    Tem = T0
    # 冷却速度
    cool = 0.95
    upf = 0
    # tfl,tff,tfc = computer_AllT(d,w,pre,m,n,tasks,Xc,Xf,Xm)
    total_time = 1000
    efl,eff,efc = computer_AllE(d,w,pre,m,n,tasks,Xc,Xf,Xm)
    minE = computer_totale(Xm, Xf, Xc, efl, eff, efc, m, n)

    '开始退火过程'
    while Tem > 0.1 and total_time > buget:
        tmptotaltime = 10000
        tmpupf = -1
        tmpE = -1
        while tmptotaltime > buget or tmpupf < 0 or Tem > 0.1:
            # print("tmptotaltime:\t", tmptotaltime)
            # print("tmpupf:\t", tmpupf)
            tmpXc = Xc.copy()
            tmpXf = Xf.copy()
            tmpXm = Xm.copy()
            '随机得到一个任务'
            tmptask = random.randint(0, n - 1)

            '随机选择任务在本地、fog、cloud执行'
            tmpChange = random.randint(1, 3)

            if tmpChange == 1:
                tmpXm[m - 1][tmptask] = 1
                tmpXf[m - 1][tmptask] = 0
                tmpXc[m - 1][tmptask] = 0
            elif tmpChange == 2:
                tmpXm[m - 1][tmptask] = 0
                tmpXf[m - 1][tmptask] = 1
                tmpXc[m - 1][tmptask] = 0
            else:
                tmpXm[m - 1][tmptask] = 0
                tmpXf[m - 1][tmptask] = 0
                tmpXc[m - 1][tmptask] = 1

            '计算能量消耗、upf、任务执行时间'
            tmpefl, tmpeff, tmpefc = computer_AllE(d, w, pre, m, n, tasks, tmpXc, tmpXf, tmpXm)
            tmpE = computer_totale(Xm, Xf, Xc, tmpefl, tmpeff, tmpefc, m, n)
            # tmpE = sum(tmpefl[m - 1] + tmpeff[m - 1] + tmpefc[m - 1])
            tmptfl, tmptff, tmptfc = computer_AllT(d, w, pre, m, n, tasks, tmpXc, tmpXf, tmpXm)
            # print("tmpefl:\t", tmpefl, "tmpeff:\t", tmpeff, "tmpefc:\t", tmpefc)
            tmptotaltime = computer_totaltime(Xm, Xf, Xc, tmptfl, tmptff, tmptfc, m, n)
            # tmptotaltime = sum(tmptfl[m - 1] + tmptff[m - 1] + tmptfc[m - 1])
            tmpupf = computer_Ufm(w, Xf, Xc, d)
            Tem = Tem * cool



        tmprand = random.random()
        if tmpE < minE:
            Xc = tmpXc.copy()
            Xf = tmpXf.copy()
            Xm = tmpXm.copy()
            minE = tmpE
            # print("get one")
        else:
            if tmprand <= math.exp(-((tmpE - minE) / Tem)):
                Xc = tmpXc.copy()
                Xf = tmpXf.copy()
                Xm = tmpXm.copy()
                minE = tmpE
                # print("get one")
        Tem = Tem * cool
        # print("Tem:\t", Tem
    print("模拟退火结束")
    return Xm,Xf,Xc


def generate(ans):
    tmp_ans = []
    tmp = []
    for arr in ans:
        tmp = arr[:]
        tmp.append(1)
        tmp_ans.append(tmp[:])
        tmp[len(tmp)-1] = 2
        tmp_ans.append(tmp[:])
        tmp[len(tmp) - 1] = 3
        tmp_ans.append(tmp[:])
    ans.clear()
    ans[:] = tmp_ans[:]



def generate_ans(N):
    ans = [[1], [2], [3]]
    for i in range(N-1):
        generate(ans)
        #print(ans)
        #print("\n\n")
    return ans
def plus(Xm, Xf, Xc, m, n):
    now = np.zeros(shape=n)

    for i in range(0, n):
        if Xm[m-1][i] == 1:
            now[i] = 1
        elif Xf[m-1][i] == 1:
            now[i] = 2
        else:
            now[i] = 3

def computer_totaltime(Xm, Xf, Xc, tfl, tff, tfc, m, n):
    '''
    计算总的能量消耗
    '''
    totatime = 0.0
    for i in range(0, n):
        totatime = totatime + Xm[m-1][i]*tfl[m-1][i] + Xf[m-1][i]*tff[m-1][i] + Xc[m-1][i]*tfc[m-1][i]
    return totatime
def exhaustion_method(d, w, pre, tasks, m, n, buget):
    '''
    穷举法求得策略
    :param d: 任务数据量大小的数组
    :param w: 任务计算量大小的数组
    :param pre: 任务执行的前置数组
    :param tasks: 任务的执行顺序
    :param m: 设备数量
    :param n: 任务数量
    :return:
    '''
    print("暴力法开始")
    ans = generate_ans(n)
    # print("the len of ans: ",len(ans))
    minE = 10000000
    Xm = np.zeros(shape=(m,n))
    Xf = np.zeros(shape=(m,n))
    Xc = np.zeros(shape=(m,n))
    for tmp in ans:
        tmpXm = np.zeros(shape=(m, n))
        tmpXf = np.zeros(shape=(m, n))
        tmpXc = np.zeros(shape=(m, n))

        'get the policy'
        for i in range(0, n):
            if tmp[i] == 1:
                tmpXm[m-1][i] = 1
                tmpXf[m-1][i] = 0
                tmpXc[m-1][i] = 0
            elif tmp[i] == 2:
                tmpXm[m-1][i] = 0
                tmpXf[m-1][i] = 1
                tmpXc[m-1][i] = 0
            else:
                tmpXm[m-1][i] = 0
                tmpXf[m-1][i] = 0
                tmpXc[m-1][i] = 1
        '计算所有的能量消耗、执行时间、fog节点的upf'

        # print("Xm:\t", tmpXm, "Xf:\t", tmpXf, "Xc:\t", tmpXc)
        tmpefl, tmpeff, tmpefc = computer_AllE(d, w,pre, m, n, tasks, tmpXc, tmpXf, tmpXm)
        totalE = computer_totale(tmpXm, tmpXf, tmpXc, tmpefl, tmpeff, tmpefc, m, n)
        # totalE = sum(tmpefl[m-1] + tmpeff[m-1] + tmpefc[m-1])
        tmptfl, tmptff, tmptfc = computer_AllT(d, w, pre, m, n, tasks, tmpXc, tmpXf, tmpXm)
        totalT = computer_totaltime(tmpXm, tmpXf, tmpXc, tmptfl, tmptff, tmptfc, m, n)
        tmpupf = computer_Ufm(w, tmpXf, tmpXc, d)

        # print("tmptfl:\t", tmptfl, "tmptff:\t", tmptff, "tmptfc:\t", tmptfc)



        '判断条件'
        # print("totalT:\t", totalT, "totalE:\t", totalE, "upf:\t", tmpupf)

        if totalT <= buget:
            # print("totalT:\t", totalT, "totalE:\t", totalE)
            if minE > totalE:
                Xm = tmpXm.copy()
                Xf = tmpXf.copy()
                Xc = tmpXc.copy()
                minE = totalE
                # print("get one ")
        else:
            continue
    print("暴力法结束")
    return Xm,Xf,Xc

def expLog(time,d, w, Xm, Xf, Xc, des, totaltime, totale, avg_d, avg_w, upf, buget):
    '''
    记录时延结果
	:param buget: 任务执行时间的预算
    :param d: 任务数据量大小
    :param w:  任务计算量的大小
    :param Xm:  任务在mobile上的执行策略
    :param Xf:  任务在fog上的执行策略
    :param Xc:  任务在云端的执行策略
    :param avg_d: 平均任务数据量的大小
    :param avg_w: 平均任务计算量的大小
    :param totaltime: 任务的执行时间
    :param totale: 任务总的能耗
    :param des: 实验简单描述
    :return:
    '''
    with open(des + ".txt", "w+") as file:
        file.write("数据量大小数组:\n")
        file.write(str(d))
        file.write("\n")
        file.write("计算量大小数组:\n")
        file.write(str(w))
        file.write("\n")
        file.write("Xm:\t")
        file.write(str(Xm))
        file.write("Xf:\t")
        file.write(str(Xf))
        file.write("Xc:\t")
        file.write(str(Xc))
        file.write("\n")
        file.write("运行时间:\t" + str(time) + "\n")
        file.write("任务总的执行时间:\t" + str(totaltime) + "\n")
        file.write("任务总的能耗:\t" + str(totale) + "\n")
        file.write("任务平均数据量的大小:\t" + str(avg_d) + "\n")
        file.write("任务平均计算量的大小:\t" + str(avg_w) + "\n")
        file.write("upf:\t" + str(upf) + "\n")
        file.write("buget:\t" + str(buget))

def exp(algor, d, w, pre, tasks, m, n, buget, des):
    import time
    now = time.time()
    algortime = 0
    if algor == 1:
        Xm,Xf,Xc = gainMethod(d, w, pre, m, n,tasks, buget)
        algortime = time.time() - now
    elif algor == 2:
        Xm,Xf,Xc = exhaustion_method(d,w,pre,tasks,m,n,buget)
        algortime = time.time() - now
    elif algor == 3:
        Xm,Xf,Xc = simulMethod(d,w,pre,tasks,m,n,30,buget)
        algortime = time.time() - now

    #平均任务计算量的大小
    avg_w = sum(w[m-1])/n

    # 平均任务数据量的大小
    avg_d = sum(d[m-1]) / n

    # 任务的执行时间
    tfl,tff,tfc = computer_AllT(d,w,pre,m,n,tasks,Xc,Xf,Xm)
    totaltime = computer_totaltime(Xm,Xf,Xc,tfl,tff,tfc,m,n)

    # 任务执行过程中消耗的能耗
    efl,eff,efc = computer_AllE(d,w,pre,m,n,tasks,Xc,Xf,Xm)
    totale = computer_totale(Xm,Xf,Xc,efl,eff,efc,m,n)

    # 计算法upf
    upf = computer_Ufm(w,Xf,Xc,d)

    expLog(algortime,d,w,Xm,Xf,Xc,des,totaltime,totale,avg_d,avg_w,upf, buget)

    return Xm,Xf,Xc

def randomonematric(m, n):
    import  random
    numberlist = [random.randint(1, 100) for i in range(0,n)]

    sumnumber = sum(numberlist)

    matric = np.zeros(shape=(m, n), dtype=np.float32)
    numberlist = [tmp/sumnumber for tmp in numberlist]

    for i in range(0,m):
        for j in range(0, n):
            matric[i][j] = numberlist[j]
    return matric


def get_offloading_result(task_num, formertask,workload, datasize, algor_type, buget_type):
    m = 1
    n = task_num
    tasks = [i for i in range(0, n)]
    pre = formertask
    d = [datasize]
    d = np.array(d) / (1024*1024)
    # d = np.random.random_sample(size=(m, n)) * 1
    w = [workload]

    print("the input data size is{0} the input workload is {1}".format(d, w))
    #
    # d = randomonematric(m, n) * 1 * (6 + 1) * 35
    # w = d * 23
    Xmlist = []
    Xflist = []
    Xclist = []
    bugetlist = []

    if buget_type == 1:
        buget = sum(w[m - 1])*0.8
    elif buget_type == 2:
        buget = sum(w[m-1])*0.5 # mid buget
    else:
        buget = sum(w[m - 1]) * 0.26
    # buget = (sum(w[m-1]))*0.8 # large budget

     # small buget
    print("the Budget is ", buget)

    Xm1, Xf1, Xc1 = exp(algor_type, d, w, pre, tasks, m, n, buget, "gaimmethodtest")

    offloading_policy = [-1 for i in range(task_num)]
    for i in range(task_num):
        if Xm1[m-1][i] == 1:
            offloading_policy[i] = 2
        if Xf1[m-1][i] == 1:
            offloading_policy[i] = 2
        if Xc1[m-1][i] == 1:
            offloading_policy[i] = 3
    return offloading_policy


if __name__ == "__main__":
    # task_num = 6
    # formertasklist = [[i-1] for i in range(6)]
    # output_time = [1547468409.1495132, 1547468410.7728095, 1547468411.5976126, 1547468412.0235333, 1547468412.171921,
    #                1547468412.2387455, 1547468407.1385837]
    # workload = []
    # for i in range(task_num):
    #     if i == 0:
    #         workload.append(output_time[0] - output_time[-1])
    #     else:
    #         workload.append(output_time[i] - output_time[i - 1])
    # workload = [tmp*10 for tmp in workload]
    # datasize = [224 * 224 * 3, 224 * 224 * 3, 112 * 112 * 64, 56 * 56 * 128, 28 * 28 * 256, 14 * 14 * 512,
    #             7 * 7 * 512]
    # # datasize = np.array(datasize) / (1024 * 8)
    # # print("workload is:", workload)
    # # print("datasize is:", datasize)


    # 'the resnet 50 '
    # task_num = 38
    # formertasklist = [[-1], [0], [0], [1, 2], [3], [3, 4], [5], [5, 6], [7], [7], [8, 9], [10], [10, 11],
    #                   [12], [12, 13], [14], [14, 15], [16], [16], [17, 18], [19], [19, 20], [21], [21, 22],
    #                   [23], [23, 24], [25], [25, 26], [27], [27, 28], [29], [29], [30, 31], [32], [32, 33],
    #                   [34], [34, 35], [36]]
    # output_time = [1547476899.3399606, 1547476899.7438064, 1547476901.792391, 1547476904.261936, 1547476905.6301556,
    #                1547476909.052161, 1547476910.4211972, 1547476913.9923651, 1547476915.3374557, 1547476917.6072588,
    #                1547476918.7249434, 1547476919.4683926, 1547476923.599801, 1547476930.3080227, 1547476936.5050642,
    #                1547476937.7935014, 1547476941.8549275, 1547476945.8670638, 1547476949.7265143, 1547476950.324865,
    #                1547476951.2173529, 1547476956.6322582, 1547476957.110144, 1547476958.6434438, 1547476959.0647304,
    #                1547476960.6138852, 1547476961.2274103, 1547476962.9769964, 1547476963.4895194, 1547476964.7773721,
    #                1547476965.6521354, 1547476966.621212, 1547476967.1023922, 1547476967.5535617, 1547476968.1555667,
    #                1547476968.455434, 1547476969.264589, 1547476969.6297235, 1547476897.498089]
    # workload = []
    # for i in range(task_num):
    #     if i == 0:
    #         workload.append(output_time[0] - output_time[-1])
    #     else:
    #         workload.append(output_time[i] - output_time[i - 1])
    # workload = [tmp for tmp in workload]
    # datasize = [224 * 224 * 3, 55 * 55 * 64, 55 * 55 * 64, 55 * 55 * 256 * 2, 55 * 55 * 256, 55 * 55 * 256 * 2,
    #             55 * 55 * 256, 55 * 55 * 256 * 2,
    #             55 * 55 * 256, 55 * 55 * 256, 28 * 28 * 512 * 2, 28 * 28 * 512, 28 * 28 * 512 * 2, 28 * 28 * 512,
    #             28 * 28 * 512 * 2, 28 * 28 * 512, 28 * 28 * 512 * 2,
    #             28 * 28 * 512, 28 * 28 * 512, 14 * 14 * 1024 * 2, 14 * 14 * 1024, 14 * 14 * 1024 * 2, 14 * 14 * 1024,
    #             14 * 14 * 1024 * 2, 14 * 14 * 1024, 14 * 14 * 1024 * 2,
    #             14 * 14 * 1024, 14 * 14 * 1024 * 2, 14 * 14 * 1024, 14 * 14 * 1024,7 * 7 * 2048 * 2,
    #             7 * 7 * 2048, 7 * 7 * 2048 * 2, 7 * 7 * 2048, 7 * 7 * 2048*2, 7 * 7 * 2048, 7 * 7 * 2048*2, 7 * 7 * 2048]
    #
    # algor_type = 3
    # print("the len of datasize :", len(datasize))
    # print("the len of the output time:", len(output_time))
    #
    # print(get_offloading_result(task_num, formertasklist, workload, datasize, algor_type))

    # 'openface'
    # task_num = 33
    # formertasklist = [[-1], [0], [0], [0], [0], [1, 2, 3, 4], [5], [5], [5], [5],
    #                   [6, 7, 8, 9], [10], [10], [10], [11, 12, 13], [14], [14],
    #                   [14], [14], [15, 16, 17, 18], [19], [19], [19], [20, 21, 22],
    #                   [23], [23], [23], [24, 25, 26], [27], [27], [27], [28, 29, 30],
    #                   [31]]
    # output_time = [1547473814.4659445, 1547473814.546958, 1547473814.6286523, 1547473814.7011578, 1547473814.7689452,
    #                1547473814.7926652, 1547473814.8418512, 1547473814.9159176, 1547473814.986788, 1547473815.0573804,
    #                1547473815.094309, 1547473815.1545618, 1547473815.2249172, 1547473815.288134, 1547473815.3123832,
    #                1547473815.3638773, 1547473815.4306982, 1547473815.4762897, 1547473815.522738, 1547473815.5532544,
    #                1547473815.5950744, 1547473815.6398182, 1547473815.6777437, 1547473815.6886353, 1547473815.7064416,
    #                1547473815.7321815, 1547473815.7560656, 1547473815.7691329, 1547473815.7826364, 1547473815.8065357,
    #                1547473815.8272645, 1547473815.8348973, 1547473815.8464653, 1547473807.4078662]
    # workload = []
    # for i in range(task_num):
    #     if i == 0:
    #         workload.append(output_time[0] - output_time[-1])
    #     else:
    #         workload.append(output_time[i] - output_time[i - 1])
    # workload = [tmp for tmp in workload]
    # datasize = [96*96*3, 12*12*192, 12*12*192, 12*12*192, 12*12*192, (12*12*128+12*12*32+12*12*32+12*12*64), 12*12*256, 12*12*256,
    #             12*12*256, 12*12*256, (12*12*128+12*12*64+12*12*64+12*12*64), 12*12*320, 12*12*320, 12*12*320, (6*6*256+6*6*64+6*6*320),
    #             6*6*640, 6*6*640, 6*6*640, 6*6*640, (6*6*192+6*6*64+6*6*128+6*6*256), 6*6*640, 6*6*640, 6*6*640, (3*3*256+3*3*128+3*3*640),
    #             3*3*1024, 3*3*1024, 3*3*1024, (3*3*384+3*3*96+3*3*256), 3*3*736, 3*3*736, 3*3*736, (3*3*384+3*3*96+3*3*256), 3*3*736]
    #
    # algor_type = 1
    # print("the len of datasize :", len(datasize))
    # print("the len of the output time:", len(output_time))
    #
    # print(get_offloading_result(task_num, formertasklist, workload, datasize, algor_type))


    'vgg16boostvgg19'
    task_num = 14
    formertasklist = [[-1], [0], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11, 12]]
    output_time = [1547478369.6544778, 1547478370.1986272, 1547478373.8009465, 1547478371.4987335, 1547478375.4653232,
                   1547478372.4462256, 1547478376.3034225, 1547478373.0028286, 1547478376.7409902, 1547478373.2289195,
                   1547478376.8916638, 1547478373.3098044, 1547478376.9549108, 1547478376.9635487, 1547478367.86583]
    workload = []
    for i in range(task_num):
        if i == 0:
            workload.append(output_time[0]-output_time[-1])
        else:
            workload.append(output_time[i] - max([output_time[tmp] for tmp in formertasklist[i]]))

    workload = [tmp for tmp in workload]
    datasize = [224 * 224 * 3, 224 * 224 * 3, 224 * 224 * 3, 112 * 112 * 64, 112 * 112 * 64, 56 * 56 * 128,
                56 * 56 * 128, 28 * 28 * 256, 28 * 28 * 256,
                14 * 14 * 512, 14 * 14 * 512, 7 * 7 * 512, 7 * 7 * 512, 2000]

    algor_type = 1
    print("the len of datasize :", len(datasize))
    print("the len of the output time:", len(output_time))

    print(get_offloading_result(task_num, formertasklist, workload, datasize, algor_type))

    pass












