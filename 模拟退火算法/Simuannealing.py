import numpy as np 
import matplotlib.pyplot as plt 


TEMP=10000
LAMBDA=0.999

def f(x,y,z):
    return 20*np.sin(x)+x**2+20*np.sin(y)+y**2+20*np.sin(z)+z**2

def simuannealing(func,init_position):
    current_position=np.array(init_position)
    T=TEMP
    step=1
    while step>0.00001:
        increment=(np.random.random(size=current_position.shape)-0.5)
        next_position=current_position+increment*step
        current_value=func(*current_position)
        next_value=func(*next_position)
        if current_value>next_value:
            current_position=next_position
            step=step*2
            T=T/LAMBDA
        else:
            accpet_prob=np.exp(-(next_value-current_value)/T)
            random_prob=np.random.random(size=1)
            if random_prob<accpet_prob:
                current_position=next_position
                print("Sparking WOW!")
            else:
                step=step*LAMBDA
        T=LAMBDA*T
        print("Current_T:",T)
        print("Current_position:",current_position)
    return current_position
    
def curve(x):
    return 10*np.sin(x)+x**2

if __name__=="__main__":
    init_position=[30,405,54]
    optimize=simuannealing(f,init_position)
    print(optimize)
    x_plot=np.linspace(5,-5,1000)
    y_plot=curve(x_plot)
    plt.plot(x_plot,y_plot)
    plt.vlines(optimize[0],-20,20,color='red',linestyles='--')
    plt.show()

    


    



        