import numpy as np 
from data_align import dataAlign
from lossfunction import *

def random_walk(func,result_preset:np.ndarray,align_array:np.ndarray,position_init:np.ndarray,weight:np.ndarray,step:float,epsilon:float,walk_num:int,iter_num:int)->np.ndarray:
    loss_init=lossfunc_every_value_norm_2_sqare(func,result_preset,align_array,position_init)
    walk_num_count=0
    loss_current=loss_init
    position_current=position_init
    while walk_num_count<walk_num and step>epsilon:
        k=1
        print("Current parameter:",position_current)
        print("Walkstep %s: current loss"%walk_num_count,loss_current)
        print("Walkstep %s: current step: "%walk_num_count,step)
        #save position log file
        with open("../log/position.log",'w') as f:
            f.write("Position path info:"+str(position_current)+'\n')
        #random walk 
        while k<iter_num:
            u=np.random.random(size=position_current.shape)-0.5
            position_walked=position_current+u*step*weight
            loss_walked=lossfunc_every_value_norm_2_sqare(func,result_preset,align_array,position_walked)
            #judement for loss decreased
            if loss_walked<loss_current:
                position_current=position_walked
                loss_current=loss_walked
                walk_num_count=walk_num_count+1
                step=step*2
                break
            else:
                position_current=position_current
                loss_current=loss_current
                k=k+1
        step=step/2
    return position_current
