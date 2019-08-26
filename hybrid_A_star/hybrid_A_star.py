# hybrid A* 

import numpy as np
import matplotlib.pyplot as plt

class node():
    def __init__(self, C, P):
        self.C = C
        self.N = [np.floor(C[0]), np.floor(C[1])]
        self.P = P
        
    def Set_cost(self, G, H):
        self.G = G
        self.H = H
        self.COST = self.G + self.H


if __name__ == '__main__':
    a = node([0.1,1.5,2],[0,0])
    print(a.C[0])
    print(a.N)
    a.Set_cost(44,22)
    print(a.COST)
