# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description:
'''

class operation:

    def __init__(self, operationid, operationfuction):
        self.operationid = operationid
        self.operationfunction = operationfuction

    def checkid(self, operationid):
        if self.operationid == operationid:
            return True
        else:
            return False

    def getid(self):
        return self.operationid

    def excute(self, inputdata):
        return self.operationfunction(inputdata)


class ExecuteAgent:

    # 定义函数
    def __func0__(self, input):
        tmp = input[0] + 1
        return tmp

    def __func1__(self, input):
        tmp = input[0] - 1
        return tmp

    def __func2__(self, input):
        tmp = input[0] * 2
        return tmp

    def __func3__(self, input):
        tmp = input[0] * input[1]
        return tmp

    def __func4__(self, input):
        tmp = input[0] * 0.5
        return tmp

    def __func5__(self, input):
        tmp = input[0] * input[1]
        return tmp

    def __func6__(self, input):
        tmp = input[0] + input[1]
        return tmp
    


    def __init__(self):
        self.operations = []

        operation0 = operation(0, self.__func0__)
        operation1 = operation(1, self.__func1__)
        operation2 = operation(2, self.__func2__)
        operation3 = operation(3, self.__func3__)
        operation4 = operation(4, self.__func4__)
        operation5 = operation(5, self.__func5__)
        operation6 = operation(6, self.__func6__)


        self.operations.append(operation0)
        self.operations.append(operation1)
        self.operations.append(operation2)
        self.operations.append(operation3)
        self.operations.append(operation4)
        self.operations.append(operation5)
        self.operations.append(operation6)


    def excute(self, operationid, inputdata):
        # #检查是否有操作id 检查输入数据格式
        # if int(operationid) >= len(self.operations)-1 or operationid < 0:
        #     return None
        #
        # if not isinstance(inputdata, list):
        #     return None

        return self.operations[operationid].excute(inputdata)

