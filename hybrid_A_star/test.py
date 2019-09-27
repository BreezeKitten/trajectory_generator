# %load hybrid_A_star.py
# hybrid A* 

# %load hybrid_A_star.py
# hybrid A* 

import numpy as np
import matplotlib.pyplot as plt
import math as m
import random
from map_load.map_load import load_img, rgb2gray

# Parameter
PI = m.pi
deltaT = 0.1 #time step
#V_set = [-1.5,-1,-0.5,0.5,1,1.5] # velocity sample set
#W_set = [0,PI/2,-PI/2,PI,-PI,PI/3,-PI/3] # angular velocity sample set
resX = 0.1 # resolution of X
resY = 0.1 # resolution of Y
resTH = PI/15 # resolution of theta
V_max = 3 #m/s
W_max = 2 #rad/s


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
    def __init__(self, C, P, V, W, Time_tag):
        self.C = C
        self.P = P
        self.V = V
        self.W = W
        self.Time_tag = Time_tag
        self.Id = '[' + str(int(np.round(self.C[0]/resX))) + ',' +  str(int(np.round(self.C[1]/resY))) + ',' + str(int(np.round(self.C[2]/resTH))) + ']'
        
    def Set_cost(self, G, H):
        self.G = G
        self.H = H
        self.COST = self.G + self.H

def angle_correct(angle):
    angle = m.fmod(angle, 2*PI)
    if angle < 0:
        angle = angle + 2*PI
    return angle


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
    TH = angle_correct(TH)
    
    return [X,Y,TH]


def Get_Vel_range(Vnow, acc_lim, Vmax, res):
    Vel_Range = np.arange(max(-Vmax,Vnow - acc_lim * deltaT), min(V_max,Vnow + acc_lim * deltaT), res)
    return Vel_Range

'''
calculate COST
'''
def Cost_cal(Nn, Ng, Vn, Wn, V, W):
    G = abs(Vn - V) + abs(Wn - W) + 10
    H = m.sqrt(abs(Nn.C[0] - Ng.C[0])**2 + abs(Nn.C[1] - Ng.C[1])**2) + abs(Nn.C[2] - Ng.C[2])

    return G, H

'''
Function ExpandNode try to expand node from the node now.
It chooses action from sample set, then creat new nodes, calculate the cost, add them into open set 'So' after check the nodes aren't
in close set 'Sc', the G cost is less than a older one
'''

    
def ExpandNode(N, Ng, So, Sc):
    V_set = Get_Vel_range(N.V, 20, V_max,0.1)
    W_set = Get_Vel_range(N.W, 20, W_max,0.1)
    for V in V_set:
        for W in W_set:
            C_temp = motionMode(N.C, V, W)
            temp = node(C_temp, N.Id, V, W, N.Time_tag + 1)
            G, H = Cost_cal(temp, Ng, N.V, N.W, V, W)
            temp.Set_cost(G + N.G,30*H)
            if temp.Id in Sc:
                pass
            elif temp.Id in So:
                if temp.G <= So[temp.Id].G:
                    So[temp.Id] = temp
            else:
                So[temp.Id] = temp
    #print('---')
    #for j in So:
     #   print(j, So[j].V)            
    return So


'''
Function find_min_cost will return the minimum cost node from the input set
'''

def find_min_cost(S):
    min_node = node([999,999,999],0,0,0,0)
    min_node.Set_cost(999999999999999999999999999999,9999999999999999999999999999)
    for i in S:
        if S[i].COST < min_node.COST:
            min_node = S[i]
    return min_node


'''
hybrid A* process:
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


def Get_path(S, Ns, Ng):
    Path = []
    Nn = Ng
    while Nn.Id != Ns.Id:
        #print(Nn.Id)
        Path.append(Nn.C)
        Nn = S[S[Nn.Id].P]
    Path.append(Nn.C)
    Path.reverse()
    return Path


def set_obs_old(Sc, im):
    [x_size, y_size] = np.shape(im)
    im_c = [[0 for i in range(x_size)] for j in range(y_size)]
    for i in range(0,x_size):
        for j in range(0,y_size):
            im_c[x_size-i-1][j] = im[i,j]
            if im[i,j] <= 0.9:
                for k in range(0,int(2*PI/resTH + 1)):
                    Id = '[' + str(j) + ',' +  str(x_size - i - 1) + ',' + str(k) + ']'                    
                    Sc[Id] = 'obs'
                #print(Id)
    return Sc, im_c


def image_transform(im):
    [x_size, y_size] = np.shape(im)
    im_c = [[0 for i in range(x_size)] for j in range(y_size)]
    for i in range(0,x_size):
        for j in range(0,y_size):
            im_c[j][x_size-i-1] = round(im[i,j])
    return im_c



def place_obs(im, obs,r):
    ID = [int(np.round(obs[0]/resX)),int(np.round(obs[1]/resY))]
    for i in range(r):
        for j in range(r-i):
            im[ID[0]-i][ID[1]-j] = 0
            im[ID[0]+i][ID[1]-j] = 0
            im[ID[0]-i][ID[1]+j] = 0
            im[ID[0]+i][ID[1]+j] = 0    
    return im


def set_obs(Sc,im):
    [x_size, y_size] = np.shape(im)
    im_show = [[0 for i in range(x_size)] for j in range(y_size)]
    for i in range(0,x_size):
        for j in range(0,y_size):
            im_show[j][i] = im[i][j]
            if im[i][j] <= 0.9:
                for k in range(0,int(2*PI/resTH + 1)):
                    Id = '[' + str(i) + ',' +  str(j) + ',' + str(k) + ']'                    
                    Sc[Id] = 'obs'
                #print(Id)
    return Sc, im_show               
                    
def Show_path(Path,im,fig_range,loop):
    for i in range(len(Path)):
        plt.imshow(im, cmap='gray', origin='lower')
        plt.axis(fig_range)
        if i == 0:
            plt.plot(Path[i][0]/resX,Path[i][1]/resY,'b.')
            plt.arrow(Path[i][0]/resX,Path[i][1]/resY,5*m.cos(Path[i][2]),5*m.sin(Path[i][2]))
            #plt.pause(0.05)
        elif i == len(Path) - 1:
            plt.plot(Path[i][0]/resX,Path[i][1]/resY,'g.')
            plt.arrow(Path[i][0]/resX,Path[i][1]/resY,5*m.cos(Path[i][2]),5*m.sin(Path[i][2]))
            #plt.pause(0.05)
        else:
            plt.plot(Path[i][0]/resX,Path[i][1]/resY,'r.')
            plt.arrow(Path[i][0]/resX,Path[i][1]/resY,5*m.cos(Path[i][2]),5*m.sin(Path[i][2]))
            #plt.pause(0.05)
        #plt.savefig('image/temp/'+str(i)+'.png')
    plt.savefig('image/0925/'+str(loop)+'.png')
    plt.show()


def State_Random():
    State = [10*random.random(),10*random.random(),2*PI*random.random()]
    return State     

if __name__ == '__main__':
    image = rgb2gray(load_img('map_load/map/empty.png'))
    for loop in range(0,1000):
        im_c = image_transform(image)
        Start_node = node(State_Random(),1,0,0,0)
        Start_node.Set_cost(0,0)
        Goal_node = node(State_Random(),1,0,0,0)
        Obs = State_Random()
        r = 1
        print('from',Start_node.Id,'to',Goal_node.Id,'Obs at [',Obs[0],Obs[1],'], r =',r)
        im_c = place_obs(im_c, [Obs[0],Obs[1]],r)
        Sop = {}
        Scl = {}
        Scl, im_show = set_obs(Scl, im_c)
        Sop, Scl, i = hybrid_A_star_process(Start_node,Goal_node,Sop,Scl)
        Path = Get_path(Scl,Start_node,Goal_node)
        print('Finsh',i)
        fig_range = [-20,120,-20,120]
        Show_path(Path,im_show,fig_range,loop)
