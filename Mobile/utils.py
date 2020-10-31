# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description:
'''
def getRandomId():
    import random
    return random.randint(0, 200000)

def check_task_done(applicationid):
    import os
    local_dir = r"E:\咸鱼工作\TMS_Exp\Mobile\Mobile\network"

    file_path = os.path.join(local_dir, "log_{0}.txt".format(applicationid))

    if os.path.exists(file_path):
        return True
    else:
        return False

def get_network_info_now(ip):
    import subprocess
    p = subprocess.Popen("iperf -c {0} -p 9999 -f M".format(ip), shell=True, stdout=subprocess.PIPE)
    wlan0_info = ''
    sar_list = []
    for i in iter(p.stdout.readline, b''):
        tmp = str(i.rstrip(), encoding='utf-8')
        # print(tmp)
        sar_list.append(tmp)

    wlan0_info = sar_list[-1]
    print(sar_list[-1].split()[6])
    return float(wlan0_info.split()[6])

def log_controle_info(applicationid, **kwargs):
    '''
    记录消息信息 将调度之前的信息写入到离线文件当中
    包括:
    * 应用id
    * 迁移策略
    * Budget
    * task graph type
    * network info
    :return:
    '''

    tmpapplicationid = applicationid
    tmpoffloadingpolicyid = -1
    tmpbudget = -1
    tmptaskgraphtype = -1
    tmpnetworkinfo = -1
    tmpalgor_name = "None"

    if "offloadingpolicyid" in kwargs:
        tmpoffloadingpolicyid = kwargs['offloadingpolicyid']

    if "budget" in kwargs:
        tmpbudget = kwargs['budget']

    if "taskgraphtypeid" in kwargs:
        tmptaskgraphtype = kwargs["taskgraphtypeid"]

    if "networkinfo" in kwargs:
        tmpnetworkinfo = kwargs["networkinfo"]

    if "algorname" in kwargs:
        tmpalgor_name = kwargs["algorname"]

    with open("log_controle_{0}.txt".format(tmpapplicationid), "a+") as file:
        file.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(tmpoffloadingpolicyid, tmpbudget, tmptaskgraphtype, tmpnetworkinfo, tmpalgor_name))


if __name__ == "__main__":
    get_network_info_now("10.21.23.134")