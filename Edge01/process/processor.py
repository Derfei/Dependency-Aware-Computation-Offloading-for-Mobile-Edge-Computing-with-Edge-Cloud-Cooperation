# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description:
'''
from model.models import  *
from model.record import *
from network.client import *
def processor_dojob(task):
    pass

def processor_getoffloadingpolicy():
    pass

def processor_getinternetinfo():
    pass

def processor_updateinternetinfo():
    pass

def processor_getapplicationinfo():
    pass

def produce_newtask(thistaskid, thistimecostlist, newtaskid,outputdata, requestdeviceid, applicatonid, offloadingpolicyid):
    '''
    生成新的任务
    :param thistaskid: 已经完成的任务id
    :param outputdata: 完成任务输出的大小
    :param requestdeviceid: 应用请求设备的id
    :param applicatonid: 应用编号
    :param offloadingpolicyid: 迁移策略id
    :return:
    '''
    tmprequestdeviceid = requestdeviceid # 请求设备id与上一个设备相同
    tmpapplicationid = applicatonid # 应用编号与上一个任务相同
    tmpoffloadingpolicyid = offloadingpolicyid # 调度策略与上一个任务相同
    tmptaskid = newtaskid

    # 通过查询应用信息获取该任务的操作编号
    tmpapplcation = getapplicationdict(requestdeviceid, applicatonid)

    if tmpapplcation == None:
        sendApplicationRequest(requestdeviceid, applicatonid) # 客户端发送应用请求信息
        print("由于应用信息不存在，向设备{0}发送请求应用{1}信息 更新设备信息".format(requestdeviceid, applicatonid))
    tmpapplcation = getapplicationdict(requestdeviceid, applicatonid)

    tmpinputdata = outputdata

    tmpformertask = [thistaskid]


    # 根据应用信息获得nexttask operationid
    tmptaskidlist = tmpapplcation['taskidlist']
    tmptaskidindex = 0
    for i in range(len(tmptaskidlist)):
        if int(tmptaskidlist[i]) == int(tmptaskid):
            tmptaskidindex = i
            break
    tmpnexttasklist = tmpapplcation['nexttasklist'][tmptaskidindex]
    tmpoperationid = tmpapplcation['operationidlist'][tmptaskidindex]

    tmptimecostlist = thistimecostlist

    tmptaskgraphtypeid = tmpapplcation['applicationtypeid']

    tmptask = task(tmprequestdeviceid, tmpapplicationid, tmptaskgraphtypeid, tmpoffloadingpolicyid, tmptaskid, tmpoperationid, tmpinputdata,
                   tmpformertask, tmpnexttasklist, tmptimecostlist)

    return tmptask

def gettaskFormertask(requestdeviceid, applicationid, taskid):
    '''
    获取特定任务的前置任务
    :param requestdeviceid: 应用请求设备编号
    :param applicationid: 应用编号
    :param taskid: 任务编号
    :return:
    '''
    tmpapplication = getapplicationdict(requestdeviceid, applicationid)

    if tmpapplication == None:
        sendApplicationRequest(requestdeviceid, applicationid)
    tmpapplication = getapplicationdict(requestdeviceid, applicationid)

    tmptaskidlist = tmpapplication['taskidlist']
    # tmptaskindex = lambda  i: int(tmptaskidlist[i])==int(taskid)
    tmptaskindex = 0
    for i in range(len(tmptaskidlist)):
        if int(tmptaskidlist[i])==int(taskid):
            tmptaskindex = i
            break
    return tmpapplication['formertasklist'][tmptaskindex]