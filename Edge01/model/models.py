# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description: 所有任务实体
'''
import  json
class MyEncoder(json.JSONEncoder):

    def default(self, o):
        import  numpy
        if isinstance(o, bytes):
            return str(o, encoding='utf-8')
        if isinstance(o, numpy.ndarray):
            return o.tolist()

        return json.JSONEncoder.default(o)

class task:

    def __init__(self, requestdeviceid, applicationid, taskgraphtypeid, offloadingpolicyid, taskid, operationid, inputdata, formertasklist,
                 nexttasklist, timecostlist):
        self.requestdevicdid = requestdeviceid
        self.applicationid = applicationid
        self.taskgraphtypeid = taskgraphtypeid
        self.offloadingpolicyid = offloadingpolicyid
        self.taskid = taskid
        self.operationid = operationid
        self.inputdata = inputdata
        self.formertasklist = formertasklist
        self.nexttasklist = nexttasklist
        self.timecostlist = timecostlist

    @classmethod
    def initfromdict(cls, taskdict):
        tmptask = task(taskdict['requestdeviceid'], taskdict['applicationid'], taskdict['taskgraphtypeid'],taskdict['offloadingpolicyid'], taskdict['taskid'],
                       taskdict['operationid'], taskdict['inputdata'], taskdict['formertasklist'], taskdict['nexttasklist'], taskdict['timecostlist'])
        return tmptask

    @classmethod
    def initfromstring(cls, taskstring):
        import json
        taskdict = json.loads(taskstring)
        return task.initfromdict(taskdict=taskdict)

    def todict(self):
        tmpdict = {}
        tmpdict['requestdeviceid'] = self.requestdevicdid
        tmpdict['applicationid'] = self.applicationid
        tmpdict['taskgraphtypeid'] = self.taskgraphtypeid
        tmpdict['offloadingpolicyid'] = self.offloadingpolicyid
        tmpdict['taskid'] = self.taskid
        tmpdict['operationid'] = self.operationid
        tmpdict['inputdata'] = self.inputdata
        tmpdict['formertasklist'] = self.formertasklist
        tmpdict['nexttasklist'] = self.nexttasklist
        tmpdict['timecostlist'] = self.timecostlist
        return tmpdict

    def tostring(self):
        import json
        tmpdict = self.todict()
        return json.dumps(tmpdict, ensure_ascii=True, cls=MyEncoder).encode()

class msg:
    '''
    requestdeviceid: 代表的是发送信息的设备编号
    senddeviceid: 发送的目标设备的id
    '''

    def __init__(self, requestdeviceid, senddeviceid, sendtime, sendmsgtype, sendmsgcontent):
        self.requestdeviceid = requestdeviceid
        self.senddeviceid = senddeviceid
        self.sendtime = sendtime
        self.sendmsgtype = sendmsgtype
        self.sendmsgcontent = sendmsgcontent

    @classmethod
    def initfromdict(cls, msgdict):
        tmpmsg = msg(msgdict['requestdeviceid'], msgdict['senddeviceid'], msgdict['sendtime'], msgdict['sendmsgtype'],
                     msgdict['sendmsgcontent'])
        return tmpmsg

    @classmethod
    def initfromstring(cls, msgstring):
        import json
        msgdict = json.loads(msgstring)
        return msg.initfromdict(msgdict)

    def todict(self):
        msgdict = {}
        msgdict['requestdeviceid'] = self.requestdeviceid
        msgdict['senddeviceid'] = self.senddeviceid
        msgdict['sendtime'] = self.sendtime
        msgdict['sendmsgtype'] = self.sendmsgtype
        msgdict['sendmsgcontent'] = self.sendmsgcontent

        return msgdict


    def tostring(self):
        import json
        tmpdict = self.todict()
        return json.dumps(tmpdict, ensure_ascii=True, cls=MyEncoder).encode()


class offloadingPolicy:

    def __init__(self, offloadingpolicyid, requestdeviceid, applicationid, taskid, excutedeviceid):
        self.offloadingpolicyid = offloadingpolicyid
        self.requestdeviceid = requestdeviceid
        self.applicationid = applicationid
        self.taskid = taskid
        self.excutedeviceid = excutedeviceid

    @classmethod
    def initfromdict(cls, offloadingpolicydict):
        tmpoffloadingpolicy = offloadingPolicy(offloadingpolicydict['offloadingpolicyid'], offloadingpolicydict['requestdeviceid'],
                                               offloadingpolicydict['applicationid'], offloadingpolicydict['taskid'],offloadingpolicydict['excutedeviceid'])
        return tmpoffloadingpolicy

    @classmethod
    def initfromstring(cls, offloadingpolicystring):
        import json
        tmpdict = json.loads(offloadingpolicystring)
        return offloadingPolicy.initfromdict(tmpdict)

    def todict(self):
        tmpdict = {}
        tmpdict['offloadingpolicyid'] = self.offloadingpolicyid
        tmpdict['requestdeviceid'] = self.requestdeviceid
        tmpdict['applicationid'] = self.applicationid
        tmpdict['taskid'] = self.taskid
        tmpdict['excutedeviceid'] = self.excutedeviceid

        return tmpdict

    def tostring(self):
        import json
        tmpdict = self.todict()
        return json.dumps(tmpdict, cls=MyEncoder).encode()


class application:

    def __init__(self, requestdeviceid, applicationid, applicationtypeid, taskidlist, formertasklist, nexttasklist, operationlist):
        self.requestdeviceid = requestdeviceid
        self.applicationid = applicationid
        self.applicationtypeid = applicationtypeid
        self.taskidlist = taskidlist
        self.formertasklist = formertasklist
        self.nexttasklist = nexttasklist
        self.operationlist = operationlist

    @classmethod
    def initfromdict(cls, applicationdict):
        tmpapplication = application(applicationdict['requestdeviceid'], applicationdict['applicationid'], applicationdict['applicationtypeid'],applicationdict['taskidlist'],
                                     applicationdict['formertasklist'], applicationdict['nexttasklist'], applicationdict['operationidlist'])
        return tmpapplication

    @classmethod
    def initfromstring(cls, applicationstring):
        import json
        tmpdict = json.loads(applicationstring)
        return application.initfromdict(tmpdict)

    @classmethod
    def initfromString(cls, applicationstringlines):
        # 将文本中的内转换为application对象
        firstline = applicationstringlines[0]
        requestdeviceid = firstline.split()[0]
        applicationid = firstline.split()[1]
        applicationtypeid = firstline.split()[2]
        taskidlist = []
        formertasklist = []
        nexttasklist = []
        operationidlist = []
        for line in applicationstringlines:
            taskidlist.append(int(line.split()[3]))
            formertasklist.append([int(tmp) for tmp in line.split()[4].split(',')])
            nexttasklist.append([int(tmp) for tmp in line.split()[5].split(',')])
            operationidlist.append(int(line.split()[6]))
        return application(requestdeviceid, applicationid, applicationtypeid,taskidlist, formertasklist,
                           nexttasklist, operationidlist)


    def todict(self):
        tmpdict = {}
        tmpdict['requestdeviceid'] = self.requestdeviceid
        tmpdict['applicationid'] = self.applicationid
        tmpdict['applicationtypeid'] = self.applicationtypeid
        tmpdict['taskidlist'] = self.taskidlist
        tmpdict['formertasklist']  = self.formertasklist
        tmpdict['nexttasklist'] = self.nexttasklist
        tmpdict['operationidlist'] = self.operationlist

        return tmpdict

    def tostring(self):
        import json
        tmpdict = self.todict()
        return json.dumps(tmpdict, cls=MyEncoder).encode()


class networkinfo:

    def __init__(self, deviceid,  devicetype, ip, port):
        self.deviceid = deviceid
        self.devicetype = devicetype
        self.ip = ip
        self.port = port

    @classmethod
    def initfromdict(cls, networkinfodict):
        tmpnetworkinfo = networkinfo(networkinfodict['deviceid'], networkinfodict['devicetype'],
                                     networkinfodict['ip'], networkinfodict['port'])
        return tmpnetworkinfo

    @classmethod
    def initfromstring(cls, networkinfostring):
        import json
        tmpnetworkinfodict = json.loads(networkinfostring)
        return networkinfo.initfromdict(tmpnetworkinfodict)

    @classmethod
    def initfromString(cls, networkinfoString):
        content = networkinfoString.split()
        tmpnetworkinfo = networkinfo(content[0], content[1], content[2], content[3])
        return tmpnetworkinfo

    def todict(self):
        tmpdict = {}
        tmpdict['deviceid'] = self.deviceid
        tmpdict['devicetype'] = self.devicetype
        tmpdict['ip'] = self.ip
        tmpdict['port'] = self.port
        return tmpdict

    def toString(self):
        tmpdict = self.todict()
        return str(tmpdict)

    def tostring(self):
        import json
        tmpdict = self.todict()
        return json.dumps(tmpdict, cls=MyEncoder).encode()


if __name__ == "__main__":
    pass






