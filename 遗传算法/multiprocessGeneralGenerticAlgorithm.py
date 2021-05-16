

'''仅仅需要修改适应度函数 F 即可完成对各种参数的遗传算法优化
'''
import pandas as pd
import numpy as np 
from data_align import dataAlign
from openseesHyteretic_speedup import opensees
import multiprocessing as mp
import queue

DNA_SIZE = 24
POP_SIZE = 40
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.005
N_GENERATIONS = 10
N_PROCESS=8


test_data=pd.read_excel("~/Documents/AutoDetectParameter/Test/test_data/LW1.5/final.xlsx")
DISP_TEST=test_data.iloc[:,1]
FORCE_TEST=test_data.iloc[:,2]

# def F(x, y,z,o):
#     # return 3*(1-x)**2*np.exp(-(x**2)-(y+1)**2)- 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2)- 1/3**np.exp(-(x+1)**2 - y**2)
#     return x+y+z+o



def lossfunc(position:np.ndarray,process_id)->float:
    result_preset=np.array(FORCE_TEST)
    align_array=np.array(DISP_TEST)
    disp_prediction,result_prediction=opensees(position,process_id)
    result_preset,result_prediction=dataAlign(align_array,result_preset,disp_prediction,result_prediction)
    if type(result_prediction) != np.ndarray:
        raise TypeError("func return value is not ndarray")
    else:
        #loss=sum((result_prediction-result_preset)**2*weight)
        loss=sum((result_prediction-result_preset)**2)
        return loss

def split_process(para_batch,process_id):
    split_pop_size=para_batch.shape[1]
    split_pred=np.zeros((split_pop_size))
    print("Process %s start"%process_id)
    for i in range(split_pop_size):
        single_parameter=para_batch[:,i]
        print("Process %s round %s"%(process_id,i))
        split_pred[i]= -lossfunc(single_parameter,process_id)
    return split_pred


def get_fitness(pop,scope):
    parameters = translateDNA(pop,scope)
    pred=np.zeros((POP_SIZE))
    batch_size=int(POP_SIZE/N_PROCESS)
    split_index=list(range(0,POP_SIZE,batch_size))
    split_para_set=np.zeros((len(scope),N_PROCESS,batch_size))
    for i,j in enumerate(split_index):
        if j==split_index[-1]:
            split_para_set[:,i,:]=parameters[:,j:]
        else:
            split_para_set[:,i,:]=parameters[:,split_index[i]:split_index[i+1]]
    #创建线程池
    process_pool=mp.Pool()
    #利用pool的apply_async方法，创建多个进程并进行并行计算，同时返回携带返回值的对象
    pred_zipped=[process_pool.apply_async(split_process,(split_para_set[:,i,:],i)) for i in range(N_PROCESS)]
    #利用get方法提取返回值
    pred_list=[item.get() for item in pred_zipped]
    #关闭已经使用完成的线程池
    process_pool.close()
    #对分组的计算的预测值进行组合
    pred=np.array(pred_list).flatten()
    print("Current loss for POP: \n",pred)
    #计算每个值的适应度
    return (pred - np.min(pred))+0.1 #减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)]

def translateDNA(pop,scope): #pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    num=len(scope)
    pop_binary=np.zeros(shape=(num,POP_SIZE,DNA_SIZE))
    pop_real_range=np.zeros(shape=(num,POP_SIZE))
    for i in range(num):
        pop_binary[i,:,:]=pop[:,i::num]
        pop_real_range[i,:]=pop_binary[i,:,:].dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(scope[i][1]-scope[i][0])+scope[i][0]
    
    return pop_real_range

def crossover_and_mutation(pop,num,CROSSOVER_RATE = 0.8):
    new_pop = []
    for father in pop:		#遍历种群中的每一个个体，将该个体作为父亲
        child = father		#孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:			#产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]	#再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE*num)	#随机产生交叉的点
            child[cross_points:] = mother[cross_points:]		#孩子得到位于交叉点后的母亲的基因
        mutation(child)	#每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop

def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE: 				#以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE*2)	#随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point]^1 	#将变异点的二进制为反转

def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness)/(fitness.sum()) )
    print("Selecting..................")
    return pop[idx]

def print_info(pop,scope):
    fitness = get_fitness(pop,scope)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    parameters = translateDNA(pop,scope)
    print("最优的基因型：", pop[max_fitness_index])
    print("Best_parameter_list:", parameters[:,max_fitness_index])
    return parameters[:,max_fitness_index]

def generticAlgorithm(scope_parameters):

    num_parameters=len(scope_parameters)
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE*num_parameters)) #matrix (POP_SIZE, DNA_SIZE)
    for _ in range(N_GENERATIONS):#迭代N代
        print("Now %s generation"%_)
        pop = np.array(crossover_and_mutation(pop,num_parameters,CROSSOVER_RATE))
        fitness = get_fitness(pop,scope_parameters)
        pop = select(pop, fitness) #选择生成新的种群
    return print_info(pop,scope_parameters)







