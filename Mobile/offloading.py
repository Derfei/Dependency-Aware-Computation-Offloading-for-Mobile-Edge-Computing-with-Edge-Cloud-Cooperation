# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description:
'''
from model.record import  *
from utils import *
from code_algor import get_offloading_result


def Offloading(workloadlist, datasizelist, formertasklist, nexttaskList, taskidList,
               applicationid, requestdeviceid, algor_type, buget_type):

    # 'vgg 16 test policy'
    offloadingpolicy = [2 for tmp in formertasklist]
    offloadingpolicy = get_offloading_result(len(taskidList), formertasklist, workloadlist, datasizelist, algor_type, buget_type)


    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidList)):
        tmpdict = {}
        tmpdict['taskid'] = taskidList[i]
        tmpdict['excuteddeviceid'] = offloadingpolicy[i]

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)

    return policy, offloadingpolicyid
def offloading(workloadlist, datasizelist, formertasklist, nexttaskList, taskidList,
               applicationid, requestdeviceid):
    '''
    该调度策略将前面几个任务放在了IoT端进行 其他的任务放在了其他设备进行
    :param workloadlist:
    :param datasizelist:
    :param formertasklist:
    :param nexttaskList:
    :param taskidList:
    :param offloadingdeviceList:
    :param applicationid:
    :param requestdeviceid:
    :return:
    '''
    '根据各项参数进行任务的迁移'
    # 将任务调度结果写入离线文本
    offloadingpolicy = [3 for tmp in formertasklist]



    offloadingpolicyid = getRandomId()


    policy = []
    for i in range(0, len(taskidList)):
        tmpdict = {}
        tmpdict['taskid'] = taskidList[i]
        tmpdict['excuteddeviceid'] = offloadingpolicy[i]

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid,applicationid=applicationid, offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)

    return policy, offloadingpolicyid

def offloading_by_greedy(workloadlist, datasizelist, formertasklist, nexttaskList, taskidList,
               applicationid, requestdeviceid, algor_type, budget_type):
    offloadingpolicy = [3, 3, 2, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3,
                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3,3,3,3]

    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidList)):
        tmpdict = {}
        tmpdict['taskid'] = taskidList[i]
        tmpdict['excuteddeviceid'] = offloadingpolicy[i]

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)

    return policy, offloadingpolicyid

def offloading_all_toedge(workloadlist, datasizelist, formertasklist, nexttaskList, taskidList,
               applicationid, requestdeviceid, algor_type, budget_type):
    import random
    offloadingpolicy = [2 for tmp in formertasklist]

    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidList)):
        tmpdict = {}
        tmpdict['taskid'] = taskidList[i]
        tmpdict['excuteddeviceid'] = offloadingpolicy[i]

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)

    return policy, offloadingpolicyid
def offloading_all_tocloud(workloadlist, datasizelist, formertasklist, nexttaskList, taskidList,
               applicationid, requestdeviceid, algor_type, budget_type):
    import random
    import numpy as np

    offloadingpolicy = [4 for tmp in formertasklist]

    for i in range(len(offloadingpolicy)):
        a = np.random.choice([2, 3, 4, 5, 6])
        offloadingpolicy[i] = a
    # offloadingpolicy[0] = 1

    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidList)):
        tmpdict = {}
        tmpdict['taskid'] = taskidList[i]
        tmpdict['excuteddeviceid'] = offloadingpolicy[i]

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)

    return policy, offloadingpolicyid


def offloading_all_to_mobile(workloadlist, datasizelist, formertasklist, nexttaskList, taskidList,
               applicationid, requestdeviceid, algor_type, budget_type):
    import random
    import numpy as np

    offloadingpolicy = [1 for tmp in formertasklist]
    #
    # for i in range(len(offloadingpolicy)):
    #     a = np.random.choice([1, 4, 5])
    #     offloadingpolicy[i] = a

    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidList)):
        tmpdict = {}
        tmpdict['taskid'] = taskidList[i]
        tmpdict['excuteddeviceid'] = offloadingpolicy[i]

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)

    return policy, offloadingpolicyid

def offloading_by_greedy_rtl(formertasklist, nexttaskList, taskidList,
               applicationid, requestdeviceid, algor_type, budget_type):
    # offloadingPolicy = []
    # for i in range(len(formertasklist)):
    #     offloadingPolicy.append(3)
    offloadingpolicy = [3 for tmp in formertasklist]
    offloadingpolicy[2] = 2
    offloadingpolicy[25] = 2

    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidList)):
        tmpdict = {}
        tmpdict['taskid'] = taskidList[i]
        tmpdict['excuteddeviceid'] = offloadingpolicy[i]

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)

    return policy, offloadingpolicyid

def offloading_by_greedy_rtl1(formertasklist, nexttaskList, taskidList,
               applicationid, requestdeviceid, algor_type, budget_type):
    # offloadingPolicy = []
    # for i in range(len(formertasklist)):
    #     offloadingPolicy.append(3)
    offloadingpolicy = [3 for tmp in formertasklist]
    offloadingpolicy[11] = 2
    # offloadingpolicy[25] = 2

    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidList)):
        tmpdict = {}
        tmpdict['taskid'] = taskidList[i]
        tmpdict['excuteddeviceid'] = offloadingpolicy[i]

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)

    return policy, offloadingpolicyid

def offloading_greedy_vggboostvgg(workloadlist, datasizelist, formertasklist, nexttaskList, taskidList,
               applicationid, requestdeviceid, algor_type, budget_type):
    offloadingpolicy = [2 for tmp in formertasklist]
    offloadingpolicy[1] = 3
    offloadingpolicy[3] = 3
    offloadingpolicy[5] = 3
    offloadingpolicy[6] = 3

    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidList)):
        tmpdict = {}
        tmpdict['taskid'] = taskidList[i]
        tmpdict['excuteddeviceid'] = offloadingpolicy[i]

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)

    return policy, offloadingpolicyid

def offloading_greedy_rtl1_vggboostvgg(workloadlist, datasizelist, formertasklist, nexttaskList, taskidList,
               applicationid, requestdeviceid, algor_type, budget_type):
    offloadingpolicy = [3 for tmp in formertasklist]
    offloadingpolicy[0] = 2
    offloadingpolicy[2] = 2

    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidList)):
        tmpdict = {}
        tmpdict['taskid'] = taskidList[i]
        tmpdict['excuteddeviceid'] = offloadingpolicy[i]

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)

    return policy, offloadingpolicyid


def offloading_openface_greedy_1(workloadlist, datasizelist, formertasklist, nexttaskList, taskidList,
               applicationid, requestdeviceid, algor_type, budget_type):
    offloadingpolicy = [3 for tmp in formertasklist]
    offloadingpolicy[2] = 2
    offloadingpolicy[3] = 2
    offloadingpolicy[4] = 2
    offloadingpolicy[7] = 2
    offloadingpolicy[8] = 2
    offloadingpolicy[9] = 2
    offloadingpolicy[12] = 2
    offloadingpolicy[13] = 2
    offloadingpolicy[16] = 2
    offloadingpolicy[17] = 2
    offloadingpolicy[18] = 2
    offloadingpolicy[21] = 2
    offloadingpolicy[22] = 2
    offloadingpolicy[25] = 2
    offloadingpolicy[26] = 2
    offloadingpolicy[29] = 2
    offloadingpolicy[30] = 2


    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidList)):
        tmpdict = {}
        tmpdict['taskid'] = taskidList[i]
        tmpdict['excuteddeviceid'] = offloadingpolicy[i]

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)

    return policy, offloadingpolicyid

def offloading_openface_greedyrtl_1(workloadlist, datasizelist, formertasklist, nexttaskList, taskidList,
               applicationid, requestdeviceid, algor_type, budget_type):
    offloadingpolicy = [3 for tmp in formertasklist]
    offloadingpolicy[7] = 2
    offloadingpolicy[10] = 2
    offloadingpolicy[11] = 2

    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidList)):
        tmpdict = {}
        tmpdict['taskid'] = taskidList[i]
        tmpdict['excuteddeviceid'] = offloadingpolicy[i]

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)

    return policy, offloadingpolicyid

def offloading_vggboostvgg_cms_1(workloadlist, datasizelist, formertasklist, nexttaskList, taskidList,
               applicationid, requestdeviceid, algor_type, budget_type):
    offloadingpolicy = [3 for tmp in formertasklist]
    # offloadingpolicy[1] = 2
    # offloadingpolicy[10] = 2
    # offloadingpolicy[11] = 2

    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidList)):
        tmpdict = {}
        tmpdict['taskid'] = taskidList[i]
        tmpdict['excuteddeviceid'] = offloadingpolicy[i]

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)

    return policy, offloadingpolicyid




