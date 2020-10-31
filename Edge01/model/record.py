# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description: 写入和读取离线文件
'''
recordbasedir = r"/home/derfei/Desktop/Edge/model/files/"
from .models import networkinfo
from .models import  *
import fcntl
def writeoffloadingpolicy(requestdeviceid, applicationid, offloadingpolicyid, offloadingpolicy):
    '''
    offloadingpolicy 离线保存格式为:
     offloaindpolicy_requestdeviceid_applicationid_offloadingpolicyid
    offloading: 格式为：
    offloadingpolicyid requestdeviceid applicationid, executedeviceid
    :param requestdeviceid:
    :param applicationid:
    :param offloadingpolicyid:
    :param offloadingpolicy:
    :return:
    '''
    import os
    import fcntl
    filepath = os.path.join(recordbasedir, 'offloadingpolicy_'+str(requestdeviceid)+"_"+str(applicationid)+"_"+str(offloadingpolicyid)+".txt")

    # 写入文件 覆盖式
    with open(filepath, "w+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        for policy in offloadingpolicy:
            line =  "{0}\t{1}\t{2}\t{3}\t{4}\n".format(offloadingpolicyid, requestdeviceid, applicationid, policy['taskid'], policy['excutedeviceid'])
            file.write(line)
    return

def writenetworkinfo(networkinfo_list):
    '''
    将传回的networkinfolist 数据写入文件当中
    :param networkinfo_list:
    :return:
    '''
    import os
    import json
    filepath = os.path.join(recordbasedir, "network.txt")

    with open(filepath, "w+") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        for networkinfo in networkinfo_list:
            if not isinstance(networkinfo, dict):
                networkinfo = json.loads(networkinfo)
            line = "{0}\t{1}\t{2}\t{3}\n".format(networkinfo['deviceid'], networkinfo['devicetype'],
                                                 networkinfo['ip'], networkinfo['port'])
            file.write(line)
    return


def getnetworkinfo(deviceid):
    '''
    从离线网络中获取网络信息
    :param deviceid: 如果为-1则为获取全部的网络信息 否则为获取一个网络信息
    :return: [type: networkinfo]  (type: ip, type: port)
    '''
    import os
    filepath = os.path.join(recordbasedir, "network.txt")

    if deviceid == None:
        return None, None

    with open(filepath, "r+") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        file.seek(0, os.SEEK_SET)
        lines = file.readlines()
        networkinfolist = []

        for line in lines:
            line = line.replace('\n', '')
            line = line.strip()
            if len(line) != 0:
                networkinfolist.append(networkinfo.initfromString(line).todict())

        # find the deviceid and return the url and the port
        if int(deviceid) == -1:
            deviceiplist = []
            deviceidlist = []
            devicetypelist = []
            deviceportlist = []

            for device in networkinfolist:
                deviceidlist.append(device['deviceid'])
                deviceiplist.append(device['ip'])
                devicetypelist.append(device['devicetype'])
                deviceportlist.append(device['port'])


            devicelist = [networkinfo(deviceidlist[tmp], devicetypelist[tmp], deviceiplist[tmp], deviceportlist[tmp]) for tmp in range(0, len(deviceiplist))]

            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            file.close()
            return devicelist
        else:
            for device in networkinfolist:
                if int(device['deviceid']) == int(deviceid):
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                    file.close()
                    return device['ip'], device['port']
    return None, None


def getapplicationinfo(taskid, requestdeviceid, applicationid):
    import os
    filepath  = os.path.join(recordbasedir, "applicationinfo_"+str(requestdeviceid)+"_"
                             +str(applicationid)+".txt")

    print("Begin to read the application file", filepath)
    # 获取应用信息
    try:
        with open(filepath, "r+") as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            file.seek(0, os.SEEK_SET)
            lines = file.readlines()
            tmpapplication = application.initfromString(lines)

            # 查找相应的应用
            formertasklist = None
            nexttasklist = None
            operationid = None

            tmpapplicationdict = tmpapplication.todict()
            for i, tmptaskid in enumerate(tmpapplicationdict['taskidlist']):
                if int(tmptaskid) == int(taskid):
                    formertasklist = tmpapplicationdict['formertasklist'][i]
                    nexttasklist = tmpapplicationdict['nexttasklist'][i]
                    operationid = tmpapplicationdict['operationidlist'][i]
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                    file.close()

                    return formertasklist, nexttasklist, operationid
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            file.close()
            return formertasklist, nexttasklist, operationid
    except Exception as e:
        return None, None, None

def getapplicationdict(requestdeviceid, applicationid):
    import os
    filepath = os.path.join(recordbasedir, "applicationinfo_"+str(requestdeviceid)+"_"+
                            str(applicationid)+".txt")

    # 获取全部的应用信息 不存在应用为空的情况
    try:
        with open(filepath, "r+") as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            file.seek(0,0)
            lines = file.readlines()

            tmpapplication = application.initfromString(lines)

            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            file.close()

            return tmpapplication.todict()
    except Exception as e:
        return None

def writeapplication(tmpapplication):
    '''
    将应用直接写入文件当中
    :param tmpapplication:
    :return:
    '''
    tmpapplicationdict = tmpapplication.todict()

    writeapplicationinfo(tmpapplicationdict['requestdeviceid'], tmpapplicationdict['applicationid'], tmpapplicationdict['applicationtypeid'],tmpapplicationdict['taskidlist'],
                         tmpapplicationdict['formertasklist'], tmpapplicationdict['nexttasklist'], tmpapplicationdict['operationidlist'])



def writeapplicationinfo(requestdeviceid, applicationid, applicationtypeid, taskidlist, formertaskidlist,
                         nexttasklist, operationidlist):
    '''
    写入应用信息
    :param requestdeviceid: 请求设备id
    :param applicationid:  应用id
    :param taskidlist:  任务id list
    :param formetaskidlist:  the percessortask list
    :param nextdeviceidlist: the nextdevice list
    :param operationlist:  the operation list
    :return:
    '''
    import os
    filepath = os.path.join(recordbasedir, "applicationinfo_"+str(requestdeviceid)+"_"
                            +str(applicationid)+".txt")


    with open(filepath, "w+") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        for i in range(0, len(taskidlist)):
            line = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(requestdeviceid, applicationid, applicationtypeid,
                                                      taskidlist[i], ','.join([str(tmp) for tmp in formertaskidlist[i]]),
                                                      ','.join([str(tmp) for tmp in nexttasklist[i]]), str(operationidlist[i]))
            file.write(line)
    return



def getoffloadingpolicyinfo(taskid,  requestdeviceid,  applicationid, offloadingpolicyid):
    import os
    import fcntl

    filepath = os.path.join(recordbasedir,"offloadingpolicy_"+str(requestdeviceid)+"_"+str(applicationid)
                            + "_" + str(offloadingpolicyid)+".txt")

    try:
        with open(filepath, 'r+') as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.seek(0, os.SEEK_SET)
            lines = file.readlines()

            if int(taskid) != -1:
                # 查找对应的task
                for line in lines:
                    line = line.replace('\n', '')
                    if int(line.split('\t')[3]) == int(taskid):
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                        file.close()
                        return int(line.split('\t')[4])
            else:
                # 获取全部的调度策略
                taskidlist = []
                excuteddeviceidlist = []

                for line in lines:
                    line = line.replace('\n', '')

                    taskidlist.append(line.split('\t')[3])
                    excuteddeviceidlist.append(line.split('\t')[4])

                # 构建调度策略应用
                offloadingpolicylist = []

                for i in range(0, len(taskidlist)):
                    tmpoffloadingpolciy = offloadingPolicy(offloadingpolicyid, requestdeviceid, applicationid, taskidlist[i],
                                                           excuteddeviceidlist[i])
                    offloadingpolicylist.append(tmpoffloadingpolciy)

                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                file.close()
                return offloadingpolicylist
    except Exception as e:
        print(e)
        return None

def getformertaskinfo(taskid, requestdeviceid, applicationid, offloadingpolicyid):
    '''
    这里有错误 还需要知道是谁的任务idlist
    获取前置任务的处理结果
    :param taskid: 需要查询任务id
    :param requestdeviceid: 应用请求设备id
    :param applicationid: 应用id号
    :return: 返回字典
    '''
    import os
    import json
    import numpy as np
    formertaskfilepath = os.path.join(recordbasedir,
                                      "formertaskinfo_{0}_{1}_{2}_{3}.txt".format(taskid, requestdeviceid, applicationid, offloadingpolicyid))
    try:
        with open(formertaskfilepath, 'r+') as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            taskdictlist = []
            file.seek(0, os.SEEK_SET)
            lines = file.readlines()
            for line in lines:
                line = line.replace('\n', '')
                print("The line split len is ", len(line.split('\t')))
                tmpdict = {}
                tmpdict['taskid'] = line.split('\t')[0]
                tmpdict['requestdeviceid'] = line.split('\t')[1]
                tmpdict['applicationid'] = line.split('\t')[2]
                tmpdict['offloadingpolicyid'] = line.split('\t')[3]
                tmpdict['formertaskid'] = line.split('\t')[4]
                # tmpdict['inputdata'] = list(line.split('\t')[5])
                # print("The tmp inputdata is {0} and the format is {1}".format(json.loads(line.split('\t')[5]), type(json.loads(line.split('\t')[5]))))
                tmpdict['inputdata'] = json.loads(line.split('\t')[5])
                tmpdict['timecost'] = json.loads(line.split('\t')[6])


                taskdictlist.append(tmpdict)

            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            file.close()
            return taskdictlist
    except Exception as e:
        print("There is a exception happend, when get the formertaskinfo", e)
        return None

def writeformertaskinfo(taskid, requestdeviceid, applicationid, offloadingpolicyid, taskdictlist):
    '''
    将前置任务的信息写入离线数据中
    :param taskid: 需要写入前置任务的任务id
    :param requestdeviceid: 提出应用的请求id
    :param applicationid: 应用id
    :param offloadingpolicyid: 迁移策略id
    :param taskdictlist: 任务字典列表上
    :return:
    '''
    import os
    import numpy as np
    import json
    formertaskfilepath = os.path.join(recordbasedir,
                                      "formertaskinfo_{0}_{1}_{2}_{3}.txt".format(taskid, requestdeviceid, applicationid,
                                                                              offloadingpolicyid))
    with open(formertaskfilepath, 'a+') as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        for tmp in taskdictlist:
            file.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(taskid, requestdeviceid, applicationid,offloadingpolicyid,
                                                          tmp['formertaskid'], json.dumps(tmp['inputdata']), json.dumps(tmp['timecost'])))

    return

def writeformertaskinfo_and_getformertaskinfo(taskid, requestdeviceid, applicationid, offloadingpolicyid, taskdictlist):
    '''
    write first and read next
    :param taskid:
    :param requestdeviceid:
    :param applicationid:
    :param offloadingpolicyid:
    :param taskdictlist:
    :return:
    '''
    import fcntl
    import os
    import numpy as np
    import json

    return_taskdictlist = []

    formertaskfilepath = os.path.join(recordbasedir,
                                      "formertaskinfo_{0}_{1}_{2}_{3}.txt".format(taskid, requestdeviceid,
                                                                                  applicationid,
                                                                                  offloadingpolicyid))
    with open(formertaskfilepath, 'a+') as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        for tmp in taskdictlist:
            file.write(
                "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(taskid, requestdeviceid, applicationid, offloadingpolicyid,
                                                             tmp['formertaskid'], json.dumps(tmp['inputdata']),
                                                             json.dumps(tmp['timecost'])))
        try:
            # taskdictlist = []
            file.seek(0, os.SEEK_SET)

            lines = file.readlines()
            for line in lines:
                line = line.replace('\n', '')
                # print("The line split len is ", len(line.split('\t')))
                tmpdict = {}
                tmpdict['taskid'] = line.split('\t')[0]
                tmpdict['requestdeviceid'] = line.split('\t')[1]
                tmpdict['applicationid'] = line.split('\t')[2]
                tmpdict['offloadingpolicyid'] = line.split('\t')[3]
                tmpdict['formertaskid'] = line.split('\t')[4]
                # tmpdict['inputdata'] = list(line.split('\t')[5])
                # print("The tmp inputdata is {0} and the format is {1}".format(json.loads(line.split('\t')[5]), type(json.loads(line.split('\t')[5]))))
                tmpdict['inputdata'] = json.loads(line.split('\t')[5])
                tmpdict['timecost'] = json.loads(line.split('\t')[6])

                return_taskdictlist.append(tmpdict)
        except Exception as e:
            print("There is a exception happend, when get the formertaskinfo", e)
            return_taskdictlist = None
    return return_taskdictlist

if __name__ == "__init__":
    pass
