# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description:
'''
from Sep import setTaskGraphGSep
from Sep import setBudgetSep
from Sep import test_get_time_cost
from gcc_5_9 import setTaskGraphGCC
from gcc_5_9 import setGccBudget
from gcc_5_9 import run
def exp1():
    '''
    四个任务图下，应用完成时间的比较，能耗的比较
    '''
    import pandas as pd
    import numpy as np
    data = pd.DataFrame(data=np.zeros(shape=(6, 4)), columns=['VGG', 'VGGboostVGG', 'Resnet50'
                                                              , 'Openface'])

    'vgg'
    setTaskGraphGSep(1)
    setBudgetSep(50)
    sep_time_1, sep_cost_1, herms_time_1, herms_cost_1 =  test_get_time_cost()

    setTaskGraphGCC(1)
    setGccBudget(50)
    _, _, gcc_time_1, gcc_cost_1  = run()

    'vggboostvgg'
    setTaskGraphGSep(2)
    setBudgetSep(50)
    sep_time_2, sep_cost_2, herms_time_2, herms_cost_2 = test_get_time_cost()

    setTaskGraphGCC(2)
    setGccBudget(50)
    _, _, gcc_time_2, gcc_cost_2 = run()

    'resnet50'
    setTaskGraphGSep(3)
    setBudgetSep(50)
    sep_time_3, sep_cost_3, herms_time_3, herms_cost_3 = test_get_time_cost()

    setTaskGraphGCC(3)
    setGccBudget(50)
    _, _, gcc_time_3, gcc_cost_3 = run()

    'Openface'
    setTaskGraphGSep(4)
    setBudgetSep(50)
    sep_time_4, sep_cost_4, herms_time_4, herms_cost_4 = test_get_time_cost()

    setTaskGraphGCC(4)
    setGccBudget(50)
    _, _, gcc_time_4, gcc_cost_4 = run()

    data.loc[0, :] = [sep_time_1, sep_time_2, sep_time_3, sep_time_4]
    data.loc[1, :] = [herms_time_1, herms_time_2, herms_time_3, herms_time_4]
    data.loc[2, :] = [gcc_time_1, gcc_time_2, gcc_time_3, gcc_time_4]
    data.loc[3, :] = [sep_cost_1, sep_cost_2, sep_cost_3, sep_cost_4]
    data.loc[4, :] = [herms_cost_1, herms_cost_2, herms_cost_3, herms_cost_4]
    data.loc[5, :] = [gcc_cost_1, gcc_cost_2, gcc_cost_3, gcc_cost_4]

    data.to_csv(index=False, path_or_buf='exp1.csv')

def exp2():
    '''
    平均用户预算下任务完成时间， 任务完成时间及消耗的变化
    :return:
    '''
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    task_graph_name_list = ['VGG', 'VGGboostVGG', 'Resnet50', 'OpenFace']
    MaxBudget = 500
    tbar = tqdm(total=MaxBudget*4)
    for task_type in range(1, 5):
        tmpData = pd.DataFrame(data=np.zeros(shape=(6, MaxBudget), dtype=np.float32), columns=[i for i in range(1, MaxBudget+1)])
        same_budget_cost =  np.zeros(shape=(3,MaxBudget), dtype=np.float32)
        same_budget_time =  np.zeros(shape=(3,MaxBudget), dtype=np.float32)
        for budget in range(1, MaxBudget+1):

            'vgg'
            setTaskGraphGSep(task_type)
            setBudgetSep(budget*0.1)
            same_budget_time[0, budget-1], same_budget_cost[0, budget-1], \
            same_budget_time[1, budget-1], same_budget_cost[1, budget-1] = test_get_time_cost()

            setTaskGraphGCC(task_type)
            setGccBudget(budget*0.1)
            _, _, same_budget_time[2, budget-1], same_budget_cost[2, budget-1]= run()
            tbar.update(1)

        algor_name_list = ['sep', 'herms', 'gcc']
        for i in range(3):
            tmpData.loc[i, :] = same_budget_time[i, :]

        for i in range(3):
            tmpData.loc[i+3, :] = same_budget_cost[i, :]

        tmpData.to_csv('{0}_exp2.csv'.format(task_graph_name_list[task_type-1]), index_label=True)
    tbar.close()



if __name__ == "__main__":
    exp1()
    exp2()