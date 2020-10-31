# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description:
'''
from schedule_algor.Sep import get_Xf_Xc, get_Xf_Xc_sep, get_Xf_Xc_hermes
from schedule_algor.gcc_5_9 import run

V1_UPLOAD = [6.43, 5.3, 5.79, 7.59]
V2_UPLOAD = [15.6, 15.34, 11.84, 8.84]
V3_UPLOAD = [0.07, 0.09, 0.51, 0.03]
V4_UPLOAD = [0.09, 4.47, 7.59, 7.2]

UPLOAD = {'v1': V1_UPLOAD, 'v2': V2_UPLOAD, 'v3': V3_UPLOAD, 'v4': V4_UPLOAD}


def setParam(algor_name='gcc'):

    def wrapper(func, *args, **kwargs):
        from schedule_algor.gcc_5_9 import setGccBudget
        from schedule_algor.gcc_5_9 import setTaskGraphGCC
        from schedule_algor.Sep import setBudgetSep
        from schedule_algor.Sep import setTaskGraphGSep
        from schedule_algor.Sep import setSepUploadRate
        from schedule_algor.gcc_5_9 import setGccUpload_rate_between_mobile_and_edge

        graph_type = 1
        budget = 1000
        upload_rate = 5

        if 'budget' in kwargs:
            budget = kwargs['budget']

        if 'graph_type' in kwargs:
            graph_type = kwargs['graph_type']

        if 'upload_rate' in kwargs:
            upload_rate = kwargs['upload_rate']


        if algor_name == "sep" or algor_name == "hermes":
            setBudgetSep(budget)
            setTaskGraphGSep(graph_type)
            setSepUploadRate(upload_rate)

        if algor_name == "gcc":
            setGccBudget(budget)
            setTaskGraphGCC(graph_type)
            setGccUpload_rate_between_mobile_and_edge(upload_rate)

        func()

    return wrapper

def measure_time(func):
    def record_time():
        import time
        begin_time = time.time()
        func()
        end_time = time.time()
        # print("用时 {0}".format(end_time-begin_time))
    return record_time



def exp1_sep(budget=1000, task_graph=1, upload_rate=5, num_image=1):
    from schedule_algor.Sep import setBudgetSep
    from schedule_algor.Sep import setTaskGraphGSep
    from schedule_algor.Sep import setSepUploadRate
    from schedule_algor.Sep import setDatasize


    setBudgetSep(budget)
    setTaskGraphGSep(task_graph)
    setSepUploadRate(upload_rate)
    setDatasize(num_image)
    Xf_sep, Xc_sep = get_Xf_Xc_sep()
    # print("Get the Xf sep is {0} Xc sep is {1} Xf hermes is {2} Xc hermes is {3}".format(
    #     Xf_sep, Xc_sep, Xf_hermes, Xc_hermes
    # ))

    return Xf_sep, Xc_sep

def exp1_hermes(budget=1000, task_graph=1, upload_rate=5, num_image=1):
    from schedule_algor.Sep import setBudgetSep
    from schedule_algor.Sep import setTaskGraphGSep
    from schedule_algor.Sep import setSepUploadRate
    from schedule_algor.Sep import setDatasize

    setBudgetSep(budget)
    setTaskGraphGSep(task_graph)
    setSepUploadRate(upload_rate)
    setDatasize(num_image)
    Xf_hermes, Xc_hermes = get_Xf_Xc_hermes()
    # print("Get the Xf sep is {0} Xc sep is {1} Xf hermes is {2} Xc hermes is {3}".format(
    #     Xf_sep, Xc_sep, Xf_hermes, Xc_hermes
    # ))

    return Xf_hermes, Xc_hermes

def exp1_gcc(budget=1000, task_graph=1, upload_rate=5, num_image=1):
    from schedule_algor.gcc_5_9 import setGccBudget
    from schedule_algor.gcc_5_9 import setTaskGraphGCC
    from schedule_algor.gcc_5_9 import setGccUpload_rate_between_mobile_and_edge, setDatasize

    setTaskGraphGCC(task_graph)
    setGccUpload_rate_between_mobile_and_edge(upload_rate)
    setGccBudget(budget)
    setDatasize(num_image)


    Xc_gcc, Xf_gcc, _, _= run()
    # print("Get the Xf gcc is {0} Xc gcc is {1} ".format(
    #     Xf_gcc, Xc_gcc
    # ))
    return Xc_gcc, Xf_gcc

def transform_X_gcc(Xf_gcc, Xc_gcc):
    import numpy as np

    offloading_policy = np.zeros(shape=(np.shape(Xc_gcc)[0]))
    Xf_gcc = np.array(Xf_gcc)
    Xc_gcc = np.array(Xc_gcc)

    'excute on edge '
    offloading_to_edge = np.where(Xf_gcc == 1)[1]
    offloading_to_edge_id = np.where(Xf_gcc == 1)[0] + 1
    device_index_dict = {1: 4, 2:5, 3: 6, 4: 2}
    for i, tmp in enumerate(offloading_to_edge):
        offloading_policy[tmp] = device_index_dict[offloading_to_edge_id[i]]

    'excute on cloud'
    offloading_to_cloud = np.where(Xc_gcc == 1)[0]
    for i, tmp in enumerate(offloading_to_cloud):
        offloading_policy[tmp] = 3

    'execute on local device'
    for i, tmp in enumerate(offloading_policy):
        if tmp == 0:
            offloading_policy[i] = 1

    return offloading_policy

def transform_X_sep(Xf_sep, Xc_sep):
    import numpy as np

    offloading_policy = np.zeros(shape=(np.shape(Xc_sep)[0]))
    Xf_sep = np.array(Xf_sep)
    Xc_sep = np.array(Xc_sep)

    'excute on edge '
    offloading_to_edge = np.where(Xf_sep == 1)[0]
    for i, tmp in enumerate(offloading_to_edge):
        offloading_policy[tmp] = 4

    'excute on cloud'
    offloading_to_cloud = np.where(Xc_sep == 1)[0]
    for i, tmp in enumerate(offloading_to_cloud):
        offloading_policy[tmp] = 3

    'execute on local device'
    for i, tmp in enumerate(offloading_policy):
        if tmp == 0:
            offloading_policy[i] = 1

    return offloading_policy

def get_application(task_grap_type):
    '''
    根据任务图构建应用 同时将应用写入离线文件当中 返回应用id
    :param task_grap_type:
    :return: 应用id
    '''
    from utils import getRandomId
    from model.record import application,writeapplication

    applicationid = getRandomId()
    taskidlist = None

    if task_grap_type == 1:
        requestdeviceid = 1
        applicationid = getRandomId()
        taskidlist = [i for i in range(8)]
        formertasklist = [[i - 1] for i in range(8)]
        nexttasklist = [[i + 1] for i in range(8)]
        nexttasklist[-1][0] = -1
        operationidlist = [i for i in range(8)]
        tmpapplication = application(requestdeviceid, applicationid, 1, taskidlist, formertasklist,
                                     nexttasklist, operationidlist)
        writeapplication(tmpapplication)  # 将应用写入离线文档中

    elif task_grap_type == 2:
        requestdeviceid = 1
        applicationid = getRandomId()
        taskidlist = [i for i in range(8)]
        formertasklist = [[-1], [0], [0], [1], [2], [3], [4], [5, 6]]
        nexttasklist = [[1, 2], [3], [4], [5], [6], [7], [7], [-1]]
        nexttasklist[-1][0] = -1
        operationidlist = [i for i in range(8)]

        tmpapplication = application(requestdeviceid, applicationid, 2, taskidlist, formertasklist,
                                     nexttasklist, operationidlist)
        writeapplication(tmpapplication)  # 将应用写入离线文档中

    elif task_grap_type == 3:
        requestdeviceid = 1
        applicationid = getRandomId()
        taskidlist = [i for i in range(38)]
        formertasklist = [[-1], [0], [0], [1, 2], [3], [3, 4], [5], [5, 6], [7], [7], [8, 9], [10], [10, 11],
                          [12], [12, 13], [14], [14, 15], [16], [16], [17, 18], [19], [19, 20], [21], [21, 22],
                          [23], [23, 24], [25], [25, 26], [27], [27, 28], [29], [29], [30, 31], [32], [32, 33],
                          [34], [34, 35], [36]]
        nexttasklist = [[1, 2], [3], [3], [4, 5], [5], [6, 7], [7], [8, 9], [10], [10], [11, 12], [12], [13, 14], [14],
                        [15, 16], [16], [17, 18], [19], [19], [20, 21], [21], [22, 23], [23], [24, 25], [25], [26, 27],
                        [27], [28, 29], [29], [30, 31], [32], [32], [33, 34], [34], [35, 36], [36], [37], [-1]]
        operationidlist = [i for i in range(38)]

        tmpapplication = application(requestdeviceid, applicationid, 3, taskidlist, formertasklist,
                                     nexttasklist, operationidlist)
        writeapplication(tmpapplication)  # 将应用写入离线文档中

    elif task_grap_type == 4:
        requestdeviceid = 1
        applicationid = getRandomId()
        taskidlist = [i for i in range(33)]
        formertasklist = [[-1], [0], [0], [0], [0], [1, 2, 3, 4], [5], [5], [5], [5],
                          [6, 7, 8, 9], [10], [10], [10], [11, 12, 13], [14], [14],
                          [14], [14], [15, 16, 17, 18], [19], [19], [19], [20, 21, 22],
                          [23], [23], [23], [24, 25, 26], [27], [27], [27], [28, 29, 30],
                          [31]]
        nexttasklist = [[1, 2, 3, 4], [5], [5], [5], [5], [6, 7, 8, 9], [10], [10],
                        [10], [10], [11, 12, 13], [14], [14], [14], [15, 16, 17, 18],
                        [19], [19], [19], [19], [20, 21, 22], [23], [23], [23],
                        [24, 25, 26], [27], [27], [27], [28, 29, 30], [31], [31], [31],
                        [32], [-1]]
        nexttasklist[-1][0] = -1
        operationidlist = [i for i in range(33)]

        tmpapplication = application(requestdeviceid, applicationid, 4, taskidlist, formertasklist,
                                     nexttasklist, operationidlist)
        writeapplication(tmpapplication)  # 将应用写入离线文档中

    return applicationid, taskidlist

def image_to_embedding(image_path):
    import  cv2
    import numpy as np


    #image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (96, 96))
    img = image[...,::-1]
    img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
    x_train = np.array([img])
    return x_train

def control_send_task(taskgraphtype, taskidlist, applicationid, offloadingpolicyid):
    import numpy as np
    from model.models import task
    import time
    from network.client import SendTask, sendTask
    import numpy as np
    import time
    from keras.preprocessing import image
    from keras.applications.imagenet_utils import preprocess_input

    double_type = 1
    local_deviceid = 1
    requestdeviceid = local_deviceid

    if taskgraphtype == 4:
        input = image_to_embedding(r'longxin.jpg')
        input = np.array([input[0] for i in range(double_type)])
        # print("the shape of input is: ", np.array(input).shape)
        time_cost_list = [[0, 0] for i in range(len(taskidlist) + 1)]
        time_cost_list[-1][0] = time.time()
        time_cost_list[-1][1] = time_cost_list[-1][0]
        tmptask = task(requestdeviceid, applicationid, taskgraphtype,offloadingpolicyid, 0, 0, np.array(input), [-1], [1, 2, 3, 4],
                       time_cost_list)

        # 发送任务
        # sendTask(objectdeviceid=firstdevice, localdeviceid=local_deviceid, newtask=tmptask)
        sendTask(local_deviceid, applicationid, offloadingpolicyid, taskgraphtype, local_deviceid, tmptask)

        # print("发送第一个任务成功 应用编号{0} 请求设备编号{1} 调度策略编号{2}".format(
        #     applicationid, requestdeviceid, offloadingpolicyid
        # ))
    elif taskgraphtype == 3:

        # 生成任务
        img_path = 'elephant.jpg'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = np.array([x, x])
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = np.array([x[0] for i in range(double_type)])
        # print("the shape of input is: ", np.array(x).shape)
        time_cost_list = [[0, 0] for i in range(len(taskidlist) + 1)]
        time_cost_list[-1][0] = time.time()
        time_cost_list[-1][1] = time_cost_list[-1][0]
        tmptask = task(requestdeviceid, applicationid, 3, offloadingpolicyid, 0, 0, np.array(x), [-1], [1, 2],
                       time_cost_list)

        # 发送任务
        sendTask(local_deviceid, applicationid, offloadingpolicyid, 0, local_deviceid, tmptask)

        # print("发送第一个任务成功 应用编号{0} 请求设备编号{1} 调度策略编号{2}".format(
        #     applicationid, requestdeviceid, offloadingpolicyid
        # ))
    elif taskgraphtype == 2:
        # 生成任务
        img_path = 'elephant.jpg'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        x = np.array([x for i in range(1)])
        x = preprocess_input(x)
        # print("the shape of input is: ", np.array(x).shape)
        time_cost_list = [[0, 0] for i in range(len(taskidlist) + 1)]
        time_cost_list[-1][0] = time.time()
        time_cost_list[-1][1] = time_cost_list[-1][0]
        tmptask = task(requestdeviceid, applicationid, 2, offloadingpolicyid, 0, 0, np.array(x), [-1], [1, 2],
                       time_cost_list)

        # 发送任务
        # sendTask(objectdeviceid=firstdevice, localdeviceid=local_deviceid, newtask=tmptask)
        sendTask(local_deviceid, applicationid, offloadingpolicyid, 0, local_deviceid, tmptask)

        # print("发送第一个任务成功 应用编号{0} 请求设备编号{1} 调度策略编号{2}".format(
        #     applicationid, requestdeviceid, offloadingpolicyid
        # ))
    elif taskgraphtype == 1:
        # 生成任务
        img_path = 'elephant.jpg'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        # x = np.array([x, x])
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = np.array([x[0] for i in range(double_type)])
        # print("the shape of input is: ", np.array(x).shape)
        time_cost_list = [[0, 0] for i in range(len(taskidlist) + 1)]
        time_cost_list[-1][0] = time.time()
        time_cost_list[-1][1] = time_cost_list[-1][0]
        tmptask = task(requestdeviceid, applicationid, 1, offloadingpolicyid, 0, 0, np.array(x), [-1], [1],
                       time_cost_list)

        # 发送任务
        # sendTask(objectdeviceid=firstdevice, localdeviceid=local_deviceid, newtask=tmptask)
        sendTask(local_deviceid, applicationid, offloadingpolicyid, 0, local_deviceid, tmptask)

        # print("发送第一个任务成功 应用编号{0} 请求设备编号{1} 调度策略编号{2}".format(
        #     applicationid, requestdeviceid, offloadingpolicyid
        # ))

# gcc
def control_exp1(budget, taskgraph_type, **kwargs):
    from utils import get_network_info_now
    from utils import  getRandomId
    from model.record import writeoffloadingpolicy
    from utils import check_task_done
    from utils import log_controle_info
    import time

    bud = budget
    taskgraph_type = taskgraph_type
    requestdeviceid = 1

    applicationid, taskidlist = get_application(taskgraph_type)

    '获取当前上传网络速率'
    # upload_rate = get_network_info_now("10.21.23.134")
    upload_rate  = 0.2
    if 'upload_rate' in kwargs:
        upload_rate = kwargs['upload_rate']

    '调用调度函数并设置参数'
    Xc_gcc, Xf_gcc = exp1_gcc(budget, taskgraph_type, upload_rate)

    '将Xf_gcc, Xc_gcc 转成迁移策略 并保存在离线文件当中'
    offloading_policy = transform_X_gcc(Xf_gcc, Xc_gcc)
    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidlist)):
        tmpdict = {}
        tmpdict['taskid'] = taskidlist[i]
        tmpdict['excuteddeviceid'] = int(offloading_policy[i])

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)


    '将应用信息保存在离线文件当中'
    log_controle_info(applicationid, budget=budget, taskgraphtypeid=taskgraph_type, networkinfo=upload_rate,
                      offloadingpolicyid=offloading_policy)

    '发送任务'
    control_send_task(taskgraph_type, taskidlist, applicationid, offloadingpolicyid)

    '确认是否已经完成任务'
    start_time = time.time()
    while not check_task_done(applicationid) and time.time() - start_time < 1000:
        pass
    if check_task_done(applicationid):
        print("应用 {0} {1}已经完成".format(applicationid, offloading_policy))
    else:
        print("应用 {0} {1}没有完成".format(applicationid, offloading_policy))

# sep
def control_exp2(budget, taskgraph_type, **kwargs):
    from utils import get_network_info_now
    from utils import  getRandomId
    from model.record import writeoffloadingpolicy
    from utils import check_task_done
    from utils import log_controle_info
    import time

    # budget = budget
    taskgraph_type = taskgraph_type
    requestdeviceid = 1

    applicationid, taskidlist = get_application(taskgraph_type)

    '获取当前上传网络速率'
    upload_rate = get_network_info_now("10.21.23.134")
    if 'upload_rate' in kwargs:
        upload_rate = kwargs['upload_rate']

    '调用调度函数并设置参数'
    Xc_sep, Xf_sep = exp1_sep(budget, taskgraph_type, upload_rate)
    # Xc_gcc, Xf_gcc = exp1_gcc(budget, taskgraph_type, upload_rate)

    '将Xf_gcc, Xc_gcc 转成迁移策略 并保存在离线文件当中'
    offloading_policy = transform_X_sep(Xf_sep[0], Xc_sep[0])
    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidlist)):
        tmpdict = {}
        tmpdict['taskid'] = taskidlist[i]
        tmpdict['excuteddeviceid'] = int(offloading_policy[i])

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)


    '将应用信息保存在离线文件当中'
    log_controle_info(applicationid, budget=budget, taskgraphtypeid=taskgraph_type, networkinfo=upload_rate,
                      offloadingpolicyid=offloading_policy)

    '发送任务'
    control_send_task(taskgraph_type, taskidlist, applicationid, offloadingpolicyid)

    '确认是否已经完成任务'
    start_time = time.time()
    while not check_task_done(applicationid) and time.time() - start_time < 350:
        pass
    if check_task_done(applicationid):
        print("应用 {0} {1}已经完成".format(applicationid, offloading_policy))
    else:
        print("应用 {0} {1}没有完成".format(applicationid, offloading_policy))

# hermes
def control_exp3(budget, taskgraph_type, **kwargs):
    from utils import get_network_info_now
    from utils import getRandomId
    from model.record import writeoffloadingpolicy
    from utils import check_task_done
    from utils import log_controle_info
    import os
    import json
    import copy


    # budget = budget
    taskgraph_type = taskgraph_type
    requestdeviceid = 1

    applicationid, taskidlist = get_application(taskgraph_type)

    '获取当前上传网络速率'
    upload_rate = get_network_info_now("10.21.23.134")
    if 'upload_rate' in kwargs:
        upload_rate = kwargs['upload_rate']

    '调用调度函数并设置参数'
    Xc_sep, Xf_sep = exp1_hermes(budget, taskgraph_type, upload_rate)
    # Xc_gcc, Xf_gcc = exp1_gcc(budget, taskgraph_type, upload_rate)

    '将Xf_gcc, Xc_gcc 转成迁移策略 并保存在离线文件当中'
    offloading_policy = transform_X_sep(Xf_sep[0], Xc_sep[0])
    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidlist)):
        tmpdict = {}
        tmpdict['taskid'] = taskidlist[i]
        tmpdict['excuteddeviceid'] = int(offloading_policy[i])

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)

    '将应用信息保存在离线文件当中'
    log_controle_info(applicationid, budget=budget, taskgraphtypeid=taskgraph_type, networkinfo=upload_rate,
                      offloadingpolicyid=offloading_policy)

    '发送任务'
    control_send_task(taskgraph_type, taskidlist, applicationid, offloadingpolicyid)

    '确认是否已经完成任务'
    while not check_task_done(applicationid):
        pass

    '返回应用完成时间 及应用完成能耗 及应用时间戳'
    application_finish_time = -1
    application_finish_cost = -1
    application_task_completing_time = None

    local_dir = r"C:\Users\derfei\Desktop\TMS_Exp\Mobile\Mobile\network"
    file_path = os.path.join(local_dir, "log_{0}.txt".format(applicationid))

    with open(file_path, 'r+') as file:
        line = file.readlines()[0]

        time_list = json.loads(line)
        application_finish_time = time_list[-1][1] - time_list[0][0]

        #computer the energy cost
        Ptx = 0.1 # 国标上限
        P0 = 1 # edge 之间传输的功耗

        Pc_edge = 65
        Pc_cloud = 65
        Pc_mobile = 3.37

        # 计算计算产生的能耗
        computation_energy_cost = 0
        for i in range(len(offloading_policy)):
            if int(offloading_policy[i]) == 1:
                computation_energy_cost += (time_list[i+1][1]-time_list[i+1][0])* Pc_mobile
            elif int(offloading_policy[i]) == 3:
                computation_energy_cost += (time_list[i+1][1]-time_list[i+1][0])* Pc_cloud
            else:
                computation_energy_cost += (time_list[i+1][1]-time_list[i+1][0])* Pc_edge

        # 通讯产生的能耗
        communication_energy_cost = 0



        application_task_completing_time = copy.deepcopy(time_list)

    print("应用 {0} 已经完成".format(applicationid))

def control_exp4(budget, taskgraph_type):
    from utils import get_network_info_now
    from utils import  getRandomId
    from model.record import writeoffloadingpolicy
    from utils import check_task_done
    from utils import log_controle_info
    import time
    import numpy as np

    # budget = budget
    taskgraph_type = taskgraph_type
    requestdeviceid = 1

    applicationid, taskidlist = get_application(taskgraph_type)

    '获取当前上传网络速率'
    upload_rate = get_network_info_now("10.21.23.134")

    '调用调度函数并设置参数'
    if taskgraph_type == 3:
        if upload_rate >= 0.7:
            Xf_gcc, Xc_gcc = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.int), \
                             np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.int)

        else:
            Xf_gcc, Xc_gcc = np.zeros(shape=(1, len(taskidlist))), np.zeros(shape=(1, len(taskidlist)))

    else:
        Xf_gcc, Xc_gcc = exp1_hermes(budget, taskgraph_type, upload_rate)

    # if taskgraph_type == 3 or taskgraph_type == 4:
    #     Xc_gcc, Xf_gcc = np.zeros(shape=(1, len(taskidlist))), np.zeros(shape=(1, len(taskidlist)))
    # else:
    #     Xc_gcc, Xf_gcc = exp1_hermes(budget, taskgraph_type, upload_rate)

    '将Xf_gcc, Xc_gcc 转成迁移策略 并保存在离线文件当中'
    offloading_policy = transform_X_sep(Xf_sep=Xf_gcc[0], Xc_sep=Xc_gcc[0])
    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidlist)):
        tmpdict = {}
        tmpdict['taskid'] = taskidlist[i]
        tmpdict['excuteddeviceid'] = int(offloading_policy[i])

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)


    '将应用信息保存在离线文件当中'
    log_controle_info(applicationid, budget=budget, taskgraphtypeid=taskgraph_type, networkinfo=upload_rate,
                      offloadingpolicyid=offloading_policy)

    '发送任务'
    control_send_task(taskgraph_type, taskidlist, applicationid, offloadingpolicyid)

    '确认是否已经完成任务'
    start_time = time.time()
    while not check_task_done(applicationid) and time.time() - start_time < 2000:
        pass
    if check_task_done(applicationid):
        print("应用 {0} {1}已经完成".format(applicationid, offloading_policy))
    else:
        print("应用 {0} {1}没有完成".format(applicationid, offloading_policy))

# hemes
def exp_1():
    from tqdm import tqdm
    tbar = tqdm(total=80)
    for task_graph in range(1, 5):
        for epoch in range(1, 20):
            # control_exp1(50, task_graph)
            if task_graph <= 2:
                control_exp3(50, task_graph)
            else:
                control_exp3(200, task_graph)
            tbar.update(1)
            # control_exp3(50, task_graph)
    tbar.close()

# sep
def exp2():
    from tqdm import tqdm
    tbar = tqdm(total=80)
    for task_graph in range(4, 5):
        for epoch in range(1, 20):
            # control_exp1(50, task_graph)
            # control_exp4(50, task_graph)
            if task_graph <= 2:
                control_exp2(50, task_graph)
            else:
                control_exp2(200, task_graph)
            tbar.update(1)
    tbar.close()

# gcc
def exp3():
    from tqdm import tqdm
    tbar = tqdm(total=80)
    for task_graph in range(2, 5):
        for epoch in range(1, 20):
            if task_graph <= 2:
                control_exp1(50, task_graph)
            else:
                control_exp1(200, task_graph)
            # control_exp4(50, task_graph)
            tbar.update(1)
            # control_exp3(50, task_graph)
    tbar.close()

def control_task_graph(task_graph_type, upload_rate, budget, algor_type, datasize):
    from utils import get_network_info_now
    from utils import getRandomId
    from model.record import writeoffloadingpolicy
    from utils import check_task_done
    from utils import log_controle_info
    import time

    requestdeviceid = 1

    applicationid, taskidlist = get_application(task_graph_type)

    # 根据算法类型调用参数
    Xc, Xf = None, None
    offloading_policy = None
    if algor_type == 'sep':
        Xc, Xf = exp1_sep(budget, task_graph_type, upload_rate, datasize)
        Xc, Xf = Xc[0], Xf[0]

        offloading_policy = transform_X_sep(Xf, Xc)
    elif algor_type == 'gcc':
        Xc, Xf = exp1_hermes(budget=budget, task_graph=task_graph_type, upload_rate=upload_rate,
                             num_image=datasize)
        Xc, Xf = Xc[0], Xf[0]
        offloading_policy = transform_X_sep(Xf, Xc)

    elif algor_type == 'hermes':
        Xc, Xf = exp1_gcc(budget=budget, task_graph=task_graph_type, upload_rate=upload_rate,
                          num_image=datasize)
        offloading_policy = transform_X_gcc(Xf, Xc)


    else:
        print("In control task graph, the algor type is wrong  not in the tlist")



    # 将Xf Xc 转换成为调度策略 并保存在离线文件当中
    offloadingpolicyid = getRandomId()

    policy = []
    for i in range(0, len(taskidlist)):
        tmpdict = {}
        tmpdict['taskid'] = taskidlist[i]
        tmpdict['excuteddeviceid'] = int(offloading_policy[i])

        policy.append(tmpdict)

    # 将迁移策略写入文件固化层
    writeoffloadingpolicy(requestdeviceid=requestdeviceid, applicationid=applicationid,
                          offloadingpolicyid=offloadingpolicyid, offloadingpolicy=policy)

    '将应用信息保存在离线文件当中'
    log_controle_info(applicationid, budget=budget, taskgraphtypeid=task_graph_type, networkinfo=upload_rate,
                      offloadingpolicyid=offloading_policy)

    '发送任务'
    control_send_task(task_graph_type, taskidlist, applicationid, offloadingpolicyid)

    '确认是否已经完成任务'
    start_time = time.time()
    while not check_task_done(applicationid) and time.time() - start_time < 600:
        pass
    if check_task_done(applicationid):
        # print("应用 {0} {1}已经完成".format(applicationid, offloading_policy))
        pass
    else:
        print("应用 {0} {1}没有完成".format(applicationid, offloading_policy))
        # exit(-1)

    return applicationid

def tmc_exp1():
    '''
    三个算法 在不同移动方式下，在四个任务上，三个的任务完成时间 完成能耗的比较
    * 先将任务进行发送， 然后将任务的返回结果进行提取，保存在单独的离线文件当中
    :return:
    '''
    from tqdm import tqdm
    import pandas as pd
    import collections
    import pandas as pd

    interation_num = 20  # 迭代次数
    col_data = collections.namedtuple('tmc_exp1_data', ['{0}_{1}_{2}'.format(algor_name, move_way, task_graph_name) for algor_name in ['sep', 'gcc', 'hermes']
                                                       for move_way in ['v1', 'v2', 'v3', 'v4']
                                                       for task_graph_name in ['vgg', 'vggboostvgg', 'resnet', 'openface']])
    tbar = tqdm(total=interation_num*1*4*4*4)
    data = []
    datalist = []
    # for inter in range(interation_num):
    #     tmpdata = col_data._make([0 for i in range(48)])
        # for i, algor_name in enumerate(['gcc', 'hermes']):
    for i, algor_name in enumerate(['sep']):
        if i == 2:
            inter_num = 1
        else:
            inter_num = interation_num
        for inter in range(inter_num):
            for j, move_way in enumerate(['v4']):
                for place in [0, 1, 2, 3]:
                    for k, task_graph_name in enumerate(['vgg', 'vggboostvgg', 'resnet', 'openface']):

                        # if k <= 2:
                        #     continue
                        # set upload rate
                        upload_rate = UPLOAD[move_way][place]

                        # set budget
                        budget = 100000

                        # set datasize
                        datasize = 1

                        applicationid = control_task_graph(task_graph_type=k+1, upload_rate=upload_rate, budget=budget,
                                           algor_type=algor_name, datasize=datasize)

                        data.append(['{0}_{1}_{2}_{3}_{4}'.format(inter, task_graph_name, move_way,
                                                                  place, algor_name), applicationid])

                        tbar.update(1)

    df = pd.DataFrame(data=data, columns=['info', 'applicationid'])
    df.to_csv('exp1_log_sep20.csv')

def tms_exp2():
    '''
       三个算法 在不同移动方式下，在四个任务上，三个的任务完成时间 完成能耗的比较
       * 先将任务进行发送， 然后将任务的返回结果进行提取，保存在单独的离线文件当中
       :return:
       '''
    from tqdm import tqdm
    import pandas as pd
    import collections
    import pandas as pd

    interation_num = 500  # 迭代次数
    col_data = collections.namedtuple('tmc_exp1_data',
                                      ['{0}_{1}_{2}'.format(algor_name, move_way, task_graph_name) for algor_name in
                                       ['sep', 'gcc', 'hermes']
                                       for move_way in ['v1', 'v2', 'v3', 'v4']
                                       for task_graph_name in ['vgg', 'vggboostvgg', 'resnet', 'openface']])
    tbar = tqdm(total=interation_num * 3 * 1 * 4 * 4 * 4 * 10)
    data = []
    datalist = []
    for inter in range(interation_num):
        tmpdata = col_data._make([0 for i in range(48)])
        for i, algor_name in enumerate(['sep', 'gcc', 'hermes']):
            for j, move_way in enumerate(['v2']):
                for place in [0, 1, 2, 3]:
                    for k, task_graph_name in enumerate(['vgg', 'vggboostvgg', 'resnet', 'openface']):
                        for tmpdata in range(10):
                            # set upload rate
                            upload_rate = UPLOAD[move_way][place]

                            # set budget
                            budget = 100000

                            # set datasize
                            datasize = tmpdata + 1

                            applicationid = control_task_graph(task_graph_type=k + 1, upload_rate=upload_rate,
                                                               budget=budget,
                                                               algor_type=algor_name, datasize=datasize)

                            data.append(['{0}_{1}_{2}_{3}_{4}'.format(inter, task_graph_name, move_way,
                                                                      place, algor_name), applicationid])

                            tbar.update(1)

    df = pd.DataFrame(data=data, columns=['info', 'applicationid'])
    df.to_csv('exp2_log.csv')



if __name__ == "__main__":
    # exp1_sep(10, 1, 0.12)
    # exp1_gcc(10, 1, 0.12)
    # exp3()
    # exp1()
    # exp2()
    # control_exp3(200, 3)
    # control_exp1(50, 3)
    # exp3()
    # control_exp2(50, 3)
    tmc_exp1()







