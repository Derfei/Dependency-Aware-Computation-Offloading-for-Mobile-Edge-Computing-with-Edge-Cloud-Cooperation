# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description:
'''
# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description: 任务信息服务器服务器
'''
import sys
sys.path.append("E:\咸鱼工作\TMS_Exp\Mobile\Mobile")
from flask import Flask
from model.record import  *
from flask.views import request
from model.record import getnetworkinfo

app  = Flask(__name__)
localdeviceid = 1

@app.route('/getOffloadingPolicy', methods=['POST'])
def getoffloadingpolicy():
    import json
    # 从数据请求中获取 应用设备id 应用id 调度策略id
    tmpoffloadingpolicydict = json.loads(request.get_data().decode('utf-8'))
    tmpoffloadingpolicydict = tmpoffloadingpolicydict['sendmsgcontent']
    applicationdeviceid = tmpoffloadingpolicydict['requestdeviceid']
    applicationid = tmpoffloadingpolicydict['applicationid']
    offloadingpolicyid = tmpoffloadingpolicydict['offloadingpolicyid']

    # app.logger.info("收到请求应用{0} 调度策略{1}".format(applicationid, offloadingpolicyid))
    # 从离线数据中获取迁移策略
    offloadingpolicylist = getoffloadingpolicyinfo(taskid=-1, requestdeviceid=applicationdeviceid, applicationid=applicationid,
                                               offloadingpolicyid=offloadingpolicyid)
    offloadingpolicylist = [tmp.todict() for tmp in offloadingpolicylist]

    # app.logger.info("从离线文档中到所有的调度策略为{0}".format(offloadingpolicylist))
    # 返回offloading策略
    return json.dumps(offloadingpolicylist, cls=MyEncoder)


@app.route('/getInternetInfo', methods=['POST'])
def getinternetinfo():
    import json
    # 从离线数据读取网络信息
    networkinfolist = getnetworkinfo(-1)

    # 返回信息
    networkinfolist = [tmp.todict() for tmp in networkinfolist]

    return json.dumps(networkinfolist, cls=MyEncoder)

@app.route('/updateInternetInfo', methods=['POST'])
def updateinternetinfo():
    import json
    # 读取网络信息
    data  = json.loads(request.get_data())
    data = data['sendmsgcontent']

    # 将网络信息写入到离线文件当中
    writenetworkinfo(data)

    return "更新成功"


@app.route('/getApplicationInfo', methods=['POST'])
def getApplicationInfo():
    import json

    data = json.loads(request.get_data().decode(encoding='utf-8'))

    # 获取本设备的设备编号
    senddeviceid = data['senddeviceid']

    # 获取需要获取的应用id
    tmpapplication = data['sendmsgcontent']
    applicationid = tmpapplication['applicationid']

    # 处理器进行处理 读取离线数据 转成json格式 进行发送
    applicationdict = getapplicationdict(senddeviceid, applicationid)

    applicationobject = application.initfromdict(applicationdict)

    return applicationobject.tostring()

def computer_energy_cost(timelist, offloadingpolicy, formertasklist):
    energy_cost = 0

    P0_network = 32
    P1_network = 64

    P0_cpu = 3500
    P1_cpu = 6400

    for i in range(len(offloadingpolicy)):
        exute_time = timelist[i][1]-timelist[i][0]

        if i == 0:
            if offloadingpolicy[i] == 2:
                energy_cost += P0_cpu*exute_time
            else:
                energy_cost += P1_cpu*exute_time
        else:
            if offloadingpolicy[i] == offloadingpolicy[i-1]:
                if offloadingpolicy[i] == 2:
                    energy_cost += P0_cpu * exute_time
                else:
                    energy_cost += P1_cpu * exute_time
            else:
                for j in formertasklist[i]:
                    network_time = timelist[i][0] - timelist[j][1]
                    if offloadingpolicy[i] == 2:
                        energy_cost = energy_cost  + P0_network * network_time
                    else:
                        energy_cost = energy_cost  + P1_network * network_time
                if offloadingpolicy[i] == 2:
                    energy_cost += P0_cpu * exute_time
                else:
                    energy_cost += P1_cpu * exute_time

    return energy_cost

@app.route('/getFinalResult', methods=['POST'])
def getFinalResult():
    import json
    import numpy as np
    from model.record import getoffloadingpolicyinfo
    from model.record import getapplicationdict
    from keras.applications.imagenet_utils import decode_predictions

    data = json.loads(request.get_data().decode('utf-8'))
    data = data['sendmsgcontent']

    tmpapplicationid = data['applicationid']
    tmprequestdeviceid = data['requestdeviceid']
    tmpoffloadingpolicyid = data['offloadingpolicyid']
    tmpinputdata = data['inputdata']
    tmptimecostlist = data['timecostlist']

    'load the offloading policy '
    offloading_policy = getoffloadingpolicyinfo(-1, tmprequestdeviceid, tmpapplicationid, tmpoffloadingpolicyid)

    'load the formertask list'
    application_dict = getapplicationdict(tmprequestdeviceid, tmpapplicationid)
    formertask_list = application_dict['formertasklist']

    workload = []
    output_time = tmptimecostlist
    # app.logger.info("The output time is:{0} ".format(output_time))
    with open("log_{0}.txt".format(tmpapplicationid), "a+") as file:
        file.write("{0}\n".format(output_time))
    for i in range(len(formertask_list)):
        workload.append(output_time[i][1]-output_time[i][0])
        # if i == 0:
        #     workload.append(output_time[i][1] - output_time[i][0])
        # else:
        #     workload.append(output_time[i] - max([output_time[tmp] for tmp in formertask_list[i]]))

    app.logger.info("The workload is: {0}".format(workload))
    with open("workload mobile {0}.txt".format(len(workload)), "a+") as file:
        file.write(','.join([str(tmp) for tmp in workload]) + '\n')
    # dataframe = pd.DataFrame(np.array(workload))
    #
    # dataframe.to_csv("workload cloud " + str(len(tmpinputdata))+ " " +  str(tmpapplicationid) + " " + ".csv",header=False, index=False)
    # app.logger.info("The shape of the output is: {0}".format(len(tmpinputdata)))

    with open("runningtime_{0}_{1}.txt".format(len(formertask_list),len(tmpinputdata)), "a+") as file:
        file.write("{0}\n".format(sum(workload)))

    # app.logger.info("应用编号{0}\t请求设备号{1}\t调度号{2}\t返回结果为{3}\t时间花费为{4} 完成任务".format(tmpapplicationid, tmprequestdeviceid,
    #                                                              tmpoffloadingpolicyid, tmpinputdata,tmptimecostlist))
    if len(tmpinputdata[0]) == 1000:
        predict = decode_predictions(np.array(tmpinputdata))
        # app.logger.info("The time cost list: {0}".format(tmptimecostlist))
        app.logger.info("应用编号{0}\t请求设备号{1}\t调度号{2}\t时间花费为{3} 完成任务 分类为 {4}".format(tmpapplicationid, tmprequestdeviceid,
                                                                          tmpoffloadingpolicyid,tmptimecostlist[-2][1] - tmptimecostlist[-1][1],
                                                                                  predict))
        # app.logger.info("时间消耗数组为{0}".format(tmptimecostlist))
        app.logger.info("消耗的总的能耗为{0}".format(computer_energy_cost(tmptimecostlist, offloading_policy, formertask_list)))

        with open('log.txt', 'a+') as file:
            file.write("应用编号{0}\t请求设备号{1}\t调度号{2}\t时间花费为{3} 完成任务 分类为 {4} 时间记录数组为 {5}\n".format(tmpapplicationid, tmprequestdeviceid,
                                                                          tmpoffloadingpolicyid,tmptimecostlist[-2][1] - tmptimecostlist[-1][1],
                                                                                  predict, tmptimecostlist))
    else:
        app.logger.info("请求数据量大小为 {4} 应用编号{0}\t请求设备号{1}\t调度号{2}\t时间花费为{3} 完成任务".format(tmpapplicationid, tmprequestdeviceid,
                                                                      tmpoffloadingpolicyid, tmptimecostlist[-2][1]-tmptimecostlist[-1][1], len(tmpinputdata)))
        # app.logger.info("时间消耗数组为{0}".format(tmptimecostlist))
        app.logger.info("消耗的总的能耗为{0}".format(computer_energy_cost(tmptimecostlist, offloading_policy, formertask_list)))
    return 'OK'

@app.route("/get_all_queue_size", methods=['POST'])
def get_all_queue_size():
    import requests
    import json

    allnetworkinfo = getnetworkinfo(-1)

    # 对每个设备发送请求任务数量的请求
    get_all_queue_size = []
    print(allnetworkinfo)
    for tmpdevice in allnetworkinfo:
        tmpdevice = tmpdevice.todict()
        try:
            if int(tmpdevice['deviceid']) != int(localdeviceid) and int(tmpdevice['deviceid']) != 3:
                data = {}
                data['type'] = 'EXECUTING_QUEUE'
                queuesize = requests.post('http://{0}:{1}/getQueuesize'.format(tmpdevice['ip'], tmpdevice['port']),
                                          data=json.dumps(data))

                if queuesize.status_code != 500:
                    tmpqueue_data = {}
                    tmpqueue_data['device_id'] = tmpdevice['deviceid']
                    # print(queuesize.text)
                    tmpqueue_data['queue_size'] = json.loads(queuesize.text)

                    get_all_queue_size.append(tmpqueue_data)
        except Exception as e:
            continue

    return_data = []
    title = ['deviceid', 'request_queue_size', 'executing_queue']
    return_data.append(title)

    for tmp_queue_size in get_all_queue_size:
        tmpdata = []
        tmpdata.append(tmp_queue_size['device_id'])
        tmpdata.append(int(tmp_queue_size['queue_size']['request_queue_size']))
        tmpdata.append(int(tmp_queue_size['queue_size']['executing_queue']))
        return_data.append(tmpdata)

    return json.dumps(return_data, cls=MyEncoder)

if __name__ == "__main__":
    print("Begin to load the agent")
    sys.path.append(r'E:\咸鱼工作\TMS_Exp\Mobile\Mobile')
    app.run(host='0.0.0.0', port=8081, debug=True, threaded=True)