# hybrid A* 

import numpy as np
import matplotlib.pyplot as plt
import math as m
from map_load.map_load import load_img, rgb2gray

# Parameter
PI = m.pi
deltaT = 0.1 #time step
V_set = [-1.5,-1,-0.5,0.5,1,1.5] # velocity sample set
W_set = [0,PI/2,-PI/2,PI,-PI,PI/3,-PI/3] # angular velocity sample set
resX = 0.1 # resolution of X
resY = 0.1 # resolution of Y
resTH = PI/15 # resolution of theta


'''
Class node contain
    C : configuration [X,Y,TH]  continuous configuration record
    P : Parent node Id
    Id : the decrete configuration of node
    G : actual cost
    H : ecpected cost
    COST : G + H (need cost parameter)
'''

class node():
    def __init__(self, C, P):
        self.C = C
        self.P = P
        self.Id = '[' + str(int(np.round(self.C[0]/resX))) + ',' +  str(int(np.round(self.C[1]/resY))) + ',' + str(int(np.round(self.C[2]/resTH))) + ']'
        
    def Set_cost(self, G, H):
        self.G = G
        self.H = H
        self.COST = self.G + self.H

'''
motionMode is used to  help agent to predict the configuration after choose some action.
'''
def motionMode(C, V, W):
    Xp = C[0]
    Yp = C[1]
    THp = C[2]
    X = Xp + V * deltaT * m.cos(THp)
    Y = Yp + V * deltaT * m.sin(THp)
    TH = THp + W * deltaT

    return [X,Y,TH]


'''
calculate COST
'''
def Cost_cal(Nn, Ng, Vn, Wn, V, W):
    G = abs(V) + abs(W)
    H = abs(Nn.C[0] - Ng.C[0]) + abs(Nn.C[1] - Ng.C[1]) + abs(Nn.C[2] - Ng.C[2])

    return G, H

'''
Function ExpandNode try to expand node from the node now.
It chooses action from sample set, then creat new nodes, calculate the cost, add them into open set 'So' after check the nodes aren't
in close set 'Sc', the G cost is less than a older one
'''

    
def ExpandNode(N, Ng, So, Sc):
    for V in V_set:
        for W in W_set:
            C_temp = motionMode(N.C, V, W)
            temp = node(C_temp, N.Id)
            G, H = Cost_cal(temp, Ng, 0, 0, V, W)
            temp.Set_cost(G + N.G,20*H)
            if temp.Id in Sc:
                a = 1
            elif temp.Id in So:
                if temp.G <= So[temp.Id].G:
                    So[temp.Id] = temp
            else:
                So[temp.Id] = temp
                
    return So


'''
Function find_min_cost will return the minimum cost node from the input set
'''

def find_min_cost(S):
    min_node = node([999,999,999],0)
    min_node.Set_cost(9999999,99999999999)
    for i in S:
        if S[i].COST < min_node.COST:
            min_node = S[i]

    return min_node


'''
A* process:
    Start by adding the start node into open set, then repeat:
    
    find the minimum cost node in open set
    expand nodes from the minimun cost node, and adding them into open set
    move the minimun cost node from open set to closed set

    repeat until the goal node in the closed set
'''

def hybrid_A_star_process(Ns, Ng, So, Sc):
    So[Ns.Id] = Ns
    i = 0
    while Ng.Id not in Sc:
        Nn = find_min_cost(So)
        So = ExpandNode(Nn, Ng, So, Sc)
        Sc[Nn.Id] = Nn
        del So[Nn.Id]
        i = i + 1
    return So, Sc, i


def Show_path(S, Ns, Ng):
    Path = []
    Nn = Ng
    while Nn.Id != Ns.Id:
        Path.append(Nn.C)
        Nn = S[S[Nn.Id].P]
        plt.plot(Path[-1][0]/resX,Path[-1][1]/resY,'ro')
        plt.arrow(Path[-1][0]/resX,Path[-1][1]/resY,1*m.cos(Path[-1][2]),1*m.sin(Path[-1][2]))
        plt.pause(0.05)
    Path.append(Nn.C)
    plt.plot(Path[-1][0]/resX,Path[-1][1]/resY,'bo')
    plt.arrow(Path[-1][0]/resX,Path[-1][1]/resY,1*m.cos(Path[-1][2]),1*m.sin(Path[-1][2]))
    plt.show()
    return Path


def set_obs(Sc, im):
    [x_size, y_size] = np.shape(im)
    im_c = [[0 for i in range(x_size)] for j in range(y_size)]
    for i in range(0,x_size-1):
        for j in range(0,y_size-1):
            im_c[x_size-i-1][j] = im[i,j]
            if im[i,j] <= 0.9:
                for k in range(0,int(2*PI/resTH + 1)):
                    Id = '[' + str(j) + ',' +  str(x_size - i - 1) + ',' + str(k) + ']'                    
                    Sc[Id] = 'obs'
                #print(Id)
    return Sc, im_c
                    
                    
    


if __name__ == '__main__':
    image = rgb2gray(load_img('map_load/map/test3.png'))
    a = node([5,5,PI/2],1)
    a.Set_cost(0,0)
    b = node([8,5,PI],1)
    Sop = {}
    Scl = {}
    Scl, im_c = set_obs(Scl, image)
    plt.imshow(im_c, cmap='Greys_r', origin='lower')
    Sop, Scl, i = hybrid_A_star_process(a,b,Sop,Scl)
    Path = Show_path(Scl,a,b)
    print('Finsh',i)
