# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description:
'''
from Util import *
def tcp_bandwidth_info(host):
    import subprocess
    cmdline = 'iperf -c {0} -t 10 -f M'.format(host)
    p = subprocess.Popen(cmdline, shell=True, stdout=subprocess.PIPE)
    cpu_info = ''
    sar_list = []
    for i in iter(p.stdout.readline, b''):
        tmp = str(i.rstrip(), encoding='utf-8')
        # print(tmp)
        sar_list.append(tmp)

    out_info = sar_list[-1]
    # print(cpu_info.split())
    return cpu_info.split()[-2]


def test_network():

    'meansure the network info'

    'move forward'

    'measure the network info'

    'move forward'

    'measure the network info'

    'move foreard'


    'back to the start place'

if __name__ == "__main__":
    tmp = tcp_bandwidth_info('10.21.23.147')
    print(tmp)


