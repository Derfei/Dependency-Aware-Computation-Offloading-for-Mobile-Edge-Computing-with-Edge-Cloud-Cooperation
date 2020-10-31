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
from flask import request, redirect, url_for
from process.processor import *
from model.record import  *
from Executer.executerDeepLearning import excuterDeepLearning
from Executer.excuteDistributedDeepLearning import  excuteDistributedDeepLearningAgent
from Executer.executeDistributedDeepLearningOneTask import excuteDistributedDeepLearningOneTaskAgent
from Executer.excuteRestnet50Onetask import excuteResnet50Onetask
from Executer.excuteVgg16CMS import excuteVgg16CMS
from Executer.excuteVgg16 import excuteVgg16
from Executer.excuter import ExecuteAgent
from Executer.excuteVgg16boostVgg19Onetask import excuteVgg16boostVgg19Onetask
from Executer.excuteResnet50 import excuteResnet50
from Executer.excuteRestnetGreedyrtl_1 import excuteResnet50Greedyrtl_1
from Executer.excuteVgg16boostVgg19 import  excuteVgg16boostVgg19
from Executer.excuteResnet50Greedyrtl import excuteResnet50Greedyrtl
from Executer.controlVggboostVggreedyrtl1 import excuteVgg16boostVgg19_greedy1
from Executer.excutefacenet_greedrtl1 import excuteDistributedDeepLearningAgent_greedyrtl1
from Executer.excuteVggboostVggCMS_1 import excuteVggboostVggCMS
# from Executer.excuteVggsa import excuteVgg16_sa
# from flask.views import request
from queue import  Queue
from threading import Thread
app  = Flask(__name__)
localdeviceid = 4
localdeviceport = 8003
max_queue_size = 2
executing_queue = Queue(max_queue_size)
request_queue = Queue()

# set the excute agent for global
print("Begin to load set the execute agent")
# excuteagent = excuterDeepLearning()
# excuteagent = excuteDistributedDeepLearningAgent()
# excuteagent = excuteDistributedDeepLearningAgent_greedyrtl1()
# excuteagent = excuteDistributedDeepLearningOneTaskAgent()
# excuteagent = excuteVgg16()
# excuteagent = excuteResnet50Onetask()
# excuteagent = excuteVgg16CMS()
# excuteagent = excuteVggboostVggCMS()
# excuteagent = excuteVgg16_sa()
# excuteagent = excuteResnet50()
# excuteagent = excuteResnet50Greedyrtl_1()
# excuteagent = excuteVgg16boostVgg19Onetask()
# excuteagent = excuteVgg16boostVgg19()
# excuteagent = excuteVgg16boostVgg19_greedy1()
# excuteagent = excuteResnet50Greedyrtl()
print("End to load set the execute agent")
def printOut(msg):
    app.logger.info(msg)

def request_dojob(taskgraphtypeid, data, executequeue):
    import requests
    try:
        requests.post(url="http://0.0.0.0:{0}/dojob".format(localdeviceport+int(taskgraphtypeid)), data=data)
        executequeue.get()
        executequeue.task_done()
    except Exception as e:
        print(e)
        pass
class Consumer(Thread):

    def __init__(self):
        Thread.__init__(self)
        self.executing_queue = executing_queue
        self.request_queue = request_queue
    def run(self):
        import time
        while True:

            if not self.executing_queue.full() and not self.request_queue.empty():
                # print("executing_queue size {0} request queue size {1}".format(self.executing_queue.qsize(),
                #                                                                self.request_queue.qsize()))
                # redirect("http://10.21.23.154:8001/do")
                # thread = Thread()
                data = self.request_queue.get()
                self.request_queue.task_done()

                self.executing_queue.put(data)

                content = json.loads(data)
                content = content['sendmsgcontent']
                thread = Thread(target=request_dojob, args=[content['taskgraphtypeid'], data, executing_queue])
                thread.start()
                # time.sleep(0.01)
                

@app.route('/dojob', methods=['POST'])
def producer():
    import json
    tmp_data = request.get_data().decode(encoding='utf-8')
    request_queue.put(tmp_data)

    content = json.loads(tmp_data)
    content = content['sendmsgcontent']
    print(
        "Get do job msf, and the content is operation : {0} applicationid: {1} offloadingpolicyid: {2} senddevice: {3} nexttasklist: {4} applicationtypeid: {5}".format(
            content['operationid'], content['applicationid'], content['offloadingpolicyid'], content['requestdeviceid'], content['nexttasklist'], content['taskgraphtypeid']
        ))
    return 'ok'


@app.route('/getQueuesize', methods=['POST', 'GET'])
def get_queue_size():
    import json

    # app.logger.info("Request get data: {0}".format(request.get_data().decode(encoding='utf-8')))
    # queue_type = json.loads(request.get_data().decode(encoding='utf-8'))
    return json.dumps({'executing_queue': executing_queue.qsize(), 'request_queue_size': request_queue.qsize()}, cls=MyEncoder)
   
if __name__ == "__main__":
    import sys
    consumer = Consumer()
    consumer.start()
    print("Begin the app run")
    sys.path.append("/home/derfei/Desktop/Edge")
    app.run(host='0.0.0.0', port=localdeviceport, debug=False, threaded=True)