'''仅仅需要修改适应度函数 F 即可完成对各种参数的遗传算法优化
'''

import numpy as np


DNA_SIZE = 24
POP_SIZE = 200
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.005
N_GENERATIONS = 100

def F(x, y,z,o):
    # return 3*(1-x)**2*np.exp(-(x**2)-(y+1)**2)- 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2)- 1/3**np.exp(-(x+1)**2 - y**2)
    return x+y+z+o

    
def get_fitness(pop,scope):
    parameters = translateDNA(pop,scope)

    pred = F(*parameters)
    print("pred:",pred,len(pred))
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
    return pop[idx]

def print_info(pop,scope):
    fitness = get_fitness(pop,scope)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    parameters = translateDNA(pop,scope)
    print("最优的基因型：", pop[max_fitness_index])
    print("Best_parameter_list:", parameters[:,max_fitness_index])

def generalGenerticAlgorithm(scope_parameters):

    num_parameters=len(scope_parameters)
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE*num_parameters)) #matrix (POP_SIZE, DNA_SIZE)
    for _ in range(N_GENERATIONS):#迭代N代
        pop = np.array(crossover_and_mutation(pop,num_parameters,CROSSOVER_RATE))
        fitness = get_fitness(pop,scope_parameters)
        pop = select(pop, fitness) #选择生成新的种群
    print_info(pop,scope_parameters)


generalGenerticAlgorithm(((-3,3),(-3,3),(-3,3),(-3,3)))



