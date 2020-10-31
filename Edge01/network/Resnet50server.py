# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description: 服务器
'''
import sys

sys.path.append("/home/derfei/Desktop/Edge")
from flask import Flask
from flask import request
from process.processor import *
from model.record import *
from Executer.excuteResnet50 import excuteResnet50


app = Flask(__name__)
localdeviceid = 4
localdeviceport = 8006

# set the excute agent for global
print("Begin to load set the execute agent")
excuteagent = excuteResnet50()
print("End to load set the execute agent")


def printOut(msg):
    app.logger.info(msg)

@app.route('/dojob', methods=['POST', 'GET'])
def dojob():
    import json
    import time
    import numpy as np

    data =  request.get_data().decode(encoding='utf-8')
    # print("Get data", data)

    data = json.loads(data)
    data = data['sendmsgcontent']

    requestdeviceid = data['requestdeviceid']
    applicationid = data['applicationid']
    offloadingpolicyid = data['offloadingpolicyid']
    taskid = data['taskid']
    operationid = data['operationid']
    inputdata = data['inputdata']
    formertasklist = data['formertasklist']
    nexttasklist = data['nexttasklist']
    timecloselist = data['timecostlist']

    # 应用信息中获取该任务的所有的前置任务
    actualformertasklist = gettaskFormertask(requestdeviceid, applicationid, taskid)
    # attention 任务结束时间这里需要进行重新设计 应该设计为任务结束的时间
    # 将任务写入前置任务中
    tmptaskdict = {}
    tmptaskdict['formertaskid'] = formertasklist[0]
    tmptaskdict['inputdata'] = inputdata
    tmptaskdict['timecost'] = timecloselist

    formertaskdictlist = writeformertaskinfo_and_getformertaskinfo(taskid=taskid, requestdeviceid=requestdeviceid,
                                                                   applicationid=applicationid,
                                                                   offloadingpolicyid=offloadingpolicyid,
                                                                   taskdictlist=[tmptaskdict])

    # 确认前置任务数据已经全部完成
    if len(actualformertasklist) != 1:

        # app.logger.info("该任务需要等待前置任务{0}完成，现在只有{1}完成".format(actualformertasklist, [tmpFormerTask['formertaskid'] for tmpFormerTask
        #                                                                  in formertaskdictlist]))
        if len(formertaskdictlist) == len(actualformertasklist):  # 任务已经全部完成 完成任务
            # 执行任务

            inputdatalist = []  # 整理输入数据按照任务id大小进行排序
            for i in range(len(formertaskdictlist) - 1):
                for j in range(len(formertaskdictlist) - i - 1):
                    if int(formertaskdictlist[j]['formertaskid']) > int(formertaskdictlist[j + 1]['formertaskid']):
                        tmp = formertaskdictlist[j]
                        formertaskdictlist[j] = formertaskdictlist[j + 1]
                        formertaskdictlist[j + 1] = tmp

            for tmp in formertaskdictlist:
                inputdatalist.append(tmp['inputdata'])

            # 合并任务完成时间
            tmpTimeCost = [tmpTime for tmpTime in timecloselist]
            for taskindex in range(len(timecloselist)):
                for tmpformertask in formertaskdictlist:

                    'debug: get cut the send time and exute time'
                    if int(tmpformertask['timecost'][taskindex][0]) != 0:
                        tmpTimeCost[taskindex] = tmpformertask['timecost'][taskindex]
                        break

            timecloselist = tmpTimeCost
            # print("前置任务不唯一，但是已经完成")
            timecloselist[int(taskid)][0] = time.time()
            print("operation id is: {0} and shape of input is {1}".format(operationid, np.shape(inputdatalist)))
            output = excuteagent.excute(operationid, inputdatalist)
            timecloselist[int(taskid)][1] = time.time()
            # app.logger.info("任务{0}已经完成 nexttasklist 为: {1} 输出为 {2}".format(taskid, nexttasklist, np.shape(output)))

            # 判断是不是最后一个任务
            if len(nexttasklist) == 1 and int(nexttasklist[0]) == -1:
                tmpnewtask = produce_newtask(taskid, timecloselist, taskid, output, requestdeviceid, applicationid,
                                             offloadingpolicyid)
                SendFinal(requestdeviceid, localdeviceid, tmpnewtask)

            else:
                # 生成新的任务
                for tmp in nexttasklist:
                    # app.logger.info("开始生成新的任务{0}".format(tmp))
                    tmpnewtask = produce_newtask(taskid, timecloselist, tmp, output, requestdeviceid, applicationid,
                                                 offloadingpolicyid)
                    # app.logger.info("生成新的任务为{0}".format(tmpnewtask.todict()))
                    SendTask(requestdeviceid, applicationid, offloadingpolicyid, tmp,
                             localdeviceid, tmpnewtask)  # 发送任务到另外的服务器

        else:  # 任务还没有全部完成
            app.logger.info("任务{0}进入等待中".format(taskid))
            pass
    else:  # 任务已经全部完成
        # 执行任务
        print("Operation {0} formertaskdictlist len is {1} applicationid is: {2} offloadingpolicyid: {3}".format(
            operationid, len(formertaskdictlist), applicationid, offloadingpolicyid
        ))

        inputdatalist = []  # 整理输入数据按照任务id大小进行排序
        for i in range(len(formertaskdictlist) - 1):
            for j in range(len(formertaskdictlist) - i - 1):
                if int(formertaskdictlist[j]['formertaskid']) > int(formertaskdictlist[j + 1]['formertaskid']):
                    tmp = formertaskdictlist[j]
                    formertaskdictlist[j] = formertaskdictlist[j + 1]
                    formertaskdictlist[j + 1] = tmp

        for tmp in formertaskdictlist:
            inputdatalist.append(tmp['inputdata'])

        # 合并任务完成时间
        tmpTimeCost = [tmpTime for tmpTime in timecloselist]
        for taskindex in range(len(timecloselist)):
            for tmpformertask in formertaskdictlist:
                'debug the time cut the time into network time and cpu time'
                if int(tmpformertask['timecost'][taskindex][0]) != 0:
                    tmpTimeCost[taskindex] = tmpformertask['timecost'][taskindex]
                    break

        timecloselist = tmpTimeCost
        timecloselist[int(taskid)][0] = time.time()
        if len(formertaskdictlist) == 1:
            inputdatalist = inputdatalist[0]
        print("operation id is: {0} and shape of input is {1}".format(operationid, np.shape(inputdatalist)))
        output = excuteagent.excute(operationid, inputdatalist)
        timecloselist[int(taskid)][1] = time.time()
        # app.logger.info("任务{0}已经完成 nexttasklist 为: {1} 输出为 {2}".format(taskid, nexttasklist, np.shape(output)))

        # 判断是不是最后一个任务
        if len(nexttasklist) == 1 and int(nexttasklist[0]) == -1:
            tmpnewtask = produce_newtask(taskid, timecloselist, taskid, output, requestdeviceid, applicationid,
                                         offloadingpolicyid)
            SendFinal(requestdeviceid, localdeviceid, tmpnewtask)
        else:
            # 生成新的任务
            for tmp in nexttasklist:
                tmpnewtask = produce_newtask(taskid, timecloselist, tmp, output, requestdeviceid, applicationid,
                                             offloadingpolicyid)

                # 根据id获取应该执行的设备
                SendTask(requestdeviceid, applicationid, offloadingpolicyid, tmp, localdeviceid,
                         tmpnewtask)  # 发送任务到另外的服务器

                # app.logger.info("从 设备 {0} 发送任务 {1} 任务内容为 {2} 到设备{3} 执行完任务 {4}".format(localdeviceid, tmp,
                #                                                                       tmpnewtask.todict(), reqUrl,
                #                                                                       taskid))

    return 'OK'

if __name__ == "__main__":
    import sys

    print("Begin the app run")
    sys.path.append("/home/derfei/Desktop/Edge")
    app.run(host='0.0.0.0', port=localdeviceport, debug=False, threaded=False)