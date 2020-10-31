# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description:
'''
import unittest
from unittest import TestCase
from model.record import *
from utils import getRandomId
class testRecord(TestCase):

    def test_writeoffloadingpolicy(self):
        writeoffloadingpolicy(1, getRandomId(), getRandomId(), [{'taskid': 0, 'excuteddeviceid': 1},
                                                                {'taskid': 1, 'excuteddeviceid': 1},
                                                                {'taskid': 2, 'excuteddeviceid': 1}])


    def test_getNetworkinfo(self):
        tmp = getnetworkinfo(1)
        print("the answer is:", tmp)


    def test_writeNetworkinfo(self):
        tmpnewworkinfo1 = networkinfo(1, 'M', "10.21.23.103", 8000)
        tmpnewworkinfo2 = networkinfo(2, 'E', "10.21.23.110", 8001)
        tmpnewworkinfo3 = networkinfo(3, 'C', "10.21.23.107", 8002)

        networkinfolist = [tmpnewworkinfo1, tmpnewworkinfo2, tmpnewworkinfo3]
        networkinfolist = [tmp.todict() for tmp in networkinfolist]

        writenetworkinfo(networkinfolist)

    def test_writeapplication(self):
        requestdeviceid = 1
        applicationid = getRandomId()
        taskidlist = [0, 1, 2, 3]
        formertasklist = [[-1], [0], [0], [1, 2]]
        nexttaskidlist = [[1, 2], [3], [3], [-1]]
        operationidlist = [0, 1, 2, 3]


        writeapplicationinfo(requestdeviceid, applicationid, taskidlist, formertasklist, nexttaskidlist,
                             operationidlist)
        print("生成application", applicationid)


    def test_getapplicationinfo(self):
        taskid = 3
        requestdeviceid = 1
        applicationid = 134298

        formetasklist, nexttasklist, operationid = getapplicationinfo(taskid, requestdeviceid,
                                                                      applicationid)

        print("When get the appilication info, the result is: {0}, {1}, {2}".format(formetasklist, nexttasklist, operationid))

    def test_getoffloadingpolicy(self):
        pass

    def test_formertaskinfo(self):
        taskid = 1
        requestdeviceid = 1
        applicationid = 16066
        offloadingPolicyid = 186643

        print("get the task {0} the offfloading device is {1}".format(1, getoffloadingpolicy(1, requestdeviceid,
                                                                                             applicationid, offloadingPolicyid)))
        print("get the task {0} the offfloading device is {1}".format(-1, getoffloadingpolicy(-1, requestdeviceid,
                                                                                             applicationid,
                                                                                             offloadingPolicyid)))


if __name__ == "__main__":
    unittest.main()


