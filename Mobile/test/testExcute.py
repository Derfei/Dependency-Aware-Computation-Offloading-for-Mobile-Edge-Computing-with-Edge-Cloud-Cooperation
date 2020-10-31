# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description: 测试执行器
'''
from executer.excuter import *
from unittest import TestCase

class testExcute(TestCase):

    def test_excuteedgent(self):
        tmpagent= ExecuteAgent()
        print("when take the operation {0}, the answer is: {1}".format(0,
                                                                       tmpagent.excute(0, [1])))
