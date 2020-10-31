# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description:
'''
from model.record import *
from network.server import printOut
LOCAL_DEVICEID = 4
from queue import Queue
def sendOffloadingpolicyRequest(requestdeviceid, applicationid, offloadingpolicyid):
    import json
    import requests
    import datetime
    '''
    发送获得迁移策略请求 获得结果 写入离线文件
    '''
    # 根据请求设备 获取ip 和端口号
    requestDeviceIp, requestDevicePort = getnetworkinfo(requestdeviceid)
    if requestDeviceIp == None:
        sendNetworkinfoRequest()
    requestDeviceIp, requestDevicePort = getnetworkinfo(requestdeviceid)

    requestDevicePort = int(requestDevicePort) + 1

    requestUrl = "http://{0}:{1}/getOffloadingPolicy".format(requestDeviceIp, requestDevicePort)

    # 发送请求
    tmpOffloadingPolicy = offloadingPolicy(offloadingpolicyid, requestdeviceid, applicationid,
                                           -1, -1)
    tmpMsg = msg(LOCAL_DEVICEID, requestdeviceid, datetime.datetime.now().__str__(),
                 'query', tmpOffloadingPolicy.todict())
    rtnMsg = requests.post(url=requestUrl, data=tmpMsg.tostring())

    #将信息写入离线文件
    # printOut("the rtnMsg is {0} 请求网络路径为 {1}".format(rtnMsg, requestUrl))
    rtnData = json.loads(rtnMsg.text)
    writeoffloadingpolicy(requestdeviceid, applicationid, offloadingpolicyid,
                          rtnData)

def sendApplicationRequest(requestdeviceid, applicationid):
    import json
    import  requests
    import datetime

    # 找到requestdeviceid 的 ip 和端口
    tmpdeviceip, tmpdeviceport = getnetworkinfo(requestdeviceid)

    if tmpdeviceip == None:
        sendNetworkinfoRequest()
    tmpdeviceip, tmpdeviceport = getnetworkinfo(requestdeviceid)

    tmpdeviceport = int(tmpdeviceport) + 1

    requrl = "http://{0}:{1}/getApplicationInfo".format(tmpdeviceip, tmpdeviceport)

    tmpapplicationinfo = application(-1, applicationid, -1, [], [], [], [])
    tmpmsg = msg(1, requestdeviceid, datetime.datetime.now().__str__(), 'qury', tmpapplicationinfo.todict())
    # 发送请求
    req = requests.post(url=requrl, data=tmpmsg.tostring())

    applicationdict = json.loads(req.text)

    writeapplicationinfo(requestdeviceid=applicationdict['requestdeviceid'], applicationid=applicationdict['applicationid'], applicationtypeid=applicationdict['applicationtypeid'],
                         taskidlist=applicationdict['taskidlist'], formertaskidlist=applicationdict['formertasklist'],
                         nexttasklist=applicationdict['nexttasklist'], operationidlist=applicationdict['operationidlist'])# 写入文件



def sendNetworkinfoRequest():
    import requests
    import json

    try:
        requrl = "http://10.21.23.103:8000/getInternetInfo"

        req = requests.post(url=requrl)
        networkinfolist  = json.loads(req)

        writenetworkinfo(networkinfolist)

        return True
    except Exception as e:
        printOut("写入网络信息返回结果出错")
        return False

def SendTask(requestdeviceid, applicationid, offloadingpolicyid,
             nexttaskid, localdeviceid, newtask):
    import threading
    thSendTask = threading.Thread(target=sendTask, args=(requestdeviceid, applicationid, offloadingpolicyid,
             nexttaskid, localdeviceid, newtask))
    thSendTask.start()

def sendTask(requestdeviceid, applicationid, offloadingpolicyid,
             nexttaskid, localdeviceid, newtask):
    import json
    import requests
    import datetime
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util import Retry


    objectdeviceid = -1

    # 根据调度信息获取执行设备  error !!!!!
    objectdeviceid = getoffloadingpolicyinfo(nexttaskid, requestdeviceid, applicationid,
                                             offloadingpolicyid)

    if objectdeviceid == None:
        sendOffloadingpolicyRequest(requestdeviceid, applicationid, offloadingpolicyid) # 请求调度信息
    objectdeviceid = getoffloadingpolicyinfo(nexttaskid, requestdeviceid, applicationid,
                                             offloadingpolicyid)

    # 获取网络信息
    tmpdeviceip, tmpdeviceport = getnetworkinfo(objectdeviceid)
    if tmpdeviceip == None:
        sendNetworkinfoRequest()
    tmpdeviceip, tmpdeviceport = getnetworkinfo(objectdeviceid)

    # 发送网络请求
    requlr = "http://{0}:{1}/dojob".format(tmpdeviceip, tmpdeviceport)
    # requlr = "{0}:{1}/dojob".format(tmpdeviceip, tmpdeviceport)
    tmpmsg = msg(localdeviceid, objectdeviceid, datetime.datetime.now().__str__(), 'dojob', newtask.todict())

    requests.post(url=requlr, data=tmpmsg.tostring(), timeout=20)
    # s = requests.Session()
    # s.mount('http://',
    #         HTTPAdapter(max_retries=Retry(total=5, method_whitelist=frozenset(['GET', 'POST']))))  # 设置 post()方法进行重访问
    # resp_post = s.post(url=requlr, data=msg.tostring())
    #
    # s.close()

    printOut("向{0}发送任务{1}成功".format(requlr, nexttaskid))

    return requlr


def sendFinal(objectdeviceid, localdeviceid, newtask):
    import json
    import requests
    import datetime

    # 获取网络信息
    tmpdeviceip, tmpdeviceport = getnetworkinfo(objectdeviceid)
    if tmpdeviceip == None:
        sendNetworkinfoRequest()
    tmpdeviceip, tmpdeviceport = getnetworkinfo(objectdeviceid)

    tmpdeviceport = int(tmpdeviceport) + 1

    # 发送网络请求
    requlr = "http://{0}:{1}/getFinalResult".format(tmpdeviceip, tmpdeviceport)
    tmpmsg = msg(localdeviceid, objectdeviceid, datetime.datetime.now().__str__(), 'finalresult', newtask.todict())

    requests.post(url=requlr, data=tmpmsg.tostring())

def SendFinal(objectdeviceid, localdeviceid, newtask):
    from threading import Thread

    th = Thread(target=sendFinal, args=[objectdeviceid, localdeviceid, newtask])
    th.start()
