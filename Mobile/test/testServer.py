# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description:0
'''
from unittest import TestCase
from model.models import  *
class testserver(TestCase):

    def test_getapplicationinfo(self):
        import json
        import requests
        import datetime
        requrl = "http://10.21.23.103:8000/getApplicationInfo"

        tmpapplicationinfo = application(1, 60294, [], [], [], [])
        tmpmsg = msg(1, 1, datetime.datetime.now().__str__(), 'query', tmpapplicationinfo.todict())

        req = requests.post(url=requrl, data=tmpmsg.tostring())

        print("测试返回的数据为: ", req.json())


    def test_updatenetworkinfo(self):
        import requests
        import datetime

        tmpnetworkinfo1 = networkinfo(1, 'M', '10.21.23.103', 8000)
        tmpnetworkinfo2 = networkinfo(2, 'E', '10.21.23.103', 8001)
        tmpnetworkinfo3 = networkinfo(3, 'C', '10.21.23.103', 8003)

        tmpnetworkinfo = [tmpnetworkinfo1.todict(), tmpnetworkinfo2.todict(), tmpnetworkinfo3.todict()]

        requrl = "http://10.21.23.103:8000/updateInternetInfo"

        tmpmsg = msg(1, 1, datetime.datetime.now().__str__(), 'update', tmpnetworkinfo)

        req = requests.post(url=requrl, data=tmpmsg.tostring())

        print("测试更新数据返回的数据为: ", req.text)

    def test_getnetworkinfo(self):

        import requests
        import datetime

        requrl = "http://10.21.23.103:8000/getInternetInfo"

        tmpmsg = msg(1, 1, datetime.datetime.now().__str__(), 'query', "")

        req = requests.post(url=requrl, data=tmpmsg.tostring())

        print("测试获取网络信息数据为: ", req.json())

    def test_getoffloadingpolicy(self):
        import requests
        import datetime

        requrl = "http://10.21.23.103:8000/getOffloadingPolicy"

        tmpoffloadingpolicy = offloadingPolicy(taskid=-1, requestdeviceid=1, applicationid=16066,
                                               offloadingpolicyid=186643, excutedeviceid=-1)

        tmpmsg = msg(1, 1, datetime.datetime.now().__str__(), "query", tmpoffloadingpolicy.todict())

        req = requests.post(url=requrl, data=tmpmsg.tostring())

        print("测试获取调度信息，调度接口返回的结果为: ", req.json())


    def test_dojob(self):

        # 构造应用

        # 进行调度

        # 根据调度结果发送任务
        pass






