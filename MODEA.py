# %% cycling 
import networkx as nx
import random
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations,chain,product
from operator import itemgetter
from initialization import Population,Individual

IND = 0
IDX = 1 
DIST = 2



class MODEA():

    def __init__(self,Gen,inertia_weight,c1,c2,graph,pop_size,pap_round=10,pm=0.1,neighbor_size=5):

        self.P = Population(graph,pop_size,pap_round,neighbor_size)
        # self.draw_current_objective_graph()
        # print('haha')
        self.Gen = Gen
        self.iw = inertia_weight
        self.c1 = c1
        self.c2 = c2
        self.pm = pm
        self.graph = graph
        self.pap_round = pap_round
        self.neighbor_size = neighbor_size
        self.pop_size = pop_size



    def cycling(self):
        t = 0
        pop = self.P.population
        for t in range(self.Gen):
            for i in range(len(pop)):
                self.gbest = random.choice(pop[i].neighbors)
                xi_pos = pop[i].get_position_as_list()

                new_v = self.calculate_new_velocity(self.gbest,pop[i])
                new_xi_pos = self.calculate_new_position(pop[i],new_v)

                if t < self.pm * self.Gen:
                    new_xi_pos = self.turbulence(new_xi_pos)

                KKM_new,RC_new = self.P.calculate_KKM_RC(new_xi_pos)

                # pop[i].update_self(new_xi_pos,new_v,KKM_new,RC_new)
                # self.set_node_position_use_list(new_xi_pos)
                # self.set_node_velocity_use_list(new_velocity)
                # self.set_KKM_RC(KKM,RC)

                self.update_neighbor_solutions(pop[i],new_xi_pos,new_v,KKM_new,RC_new)
                self.P.set_ref_point()
                self.P.update_pbest(pop[i])

            self.draw_current_objective_graph()
        
    def draw_current_objective_graph(self):
        pop = self.P.population

        KKM_result = [pop[i].RC for i in range(len(pop))]
        RC_result = [pop[i].KKM for i in range(len(pop))]
        plt.scatter(KKM_result, RC_result, marker='^', alpha=1)
        plt.xlabel("RC")
        plt.ylabel("KKM")
        plt.show()
                        

    def update_neighbor_solutions(self,indiv,new_pos,new_v,KKM_new,RC_new):

        for neighbor in indiv.neighbors:
            neibr_pos = neighbor[IND].get_position_as_list()
            KKM_neibr,RC_neibr = self.P.calculate_KKM_RC(neibr_pos)
            
            neibr_ind = neighbor[IND]
            neibr_idx = neighbor[IDX]

            g_new = max([neibr_ind.weight[0] * KKM_new-self.P.ref[0],neibr_ind.weight[1] * RC_new-self.P.ref[1]])
            g_old = max([neibr_ind.weight[0] * KKM_neibr-self.P.ref[0],neibr_ind.weight[1] * RC_neibr-self.P.ref[1]])
            if g_new < g_old:
                neibr_ind.set_node_position_use_list(new_pos)
                neibr_ind.set_node_velocity_use_list(new_v)
                neibr_ind.set_KKM_RC(KKM_neibr,RC_neibr)
            

    def sigmoid(self, V):
        Y = []
        for x in V:
            if random.random() < 1/math.exp(-x):
                Y.append(1)
            else:
                Y.append(0)
        return Y

    def XOR(self,l1,l2):
        return [i^j for i,j in zip(l1,l2)]

    def calculate_new_velocity(self,gbest,xi):
        gbest_pos = gbest[0].get_position_as_list()
        xi_pos = xi.get_position_as_list()
        xi_pbest = xi.get_pbest_as_list()
        xi_v = xi.get_velocity_as_list()

        new_v = self.sigmoid(random.random() * np.asfarray(xi_v) + self.c1 * random.random() * np.asfarray(self.XOR(xi_pbest,xi_pos)) + \
         self.c2*np.asfarray(random.random()) * np.asfarray(self.XOR(gbest_pos,xi_pos)))

        return new_v

    def get_nbest(self,node,xi):
        neighbors = [m for m in self.graph.neighbors(node)]
        if len(neighbors) > 0:
            community = dict()
            for neighbor in neighbors:
                key = xi.graph.nodes[neighbor]['position']
                community[key] = community.get(key,0)+1
            result = max(community.items(), key=itemgetter(1))[0]
        else:
            result = self.graph.nodes[node]['position']
        return result

    def calculate_new_position(self,xi,new_v):

        xi_v = new_v
        xi_pos = xi.get_position_as_list()
        new_xi_pos = []
        for k,(vi,p) in enumerate(zip(xi_v,xi_pos)):
            if vi == 0:
                new_xi_pos.append(p)
            else:
                Nbest = self.get_nbest(k,xi)
                new_xi_pos.append(Nbest)
        return new_xi_pos

    def turbulence(self,new_pos):
        xi = new_pos
        for i in range(len(self.graph.nodes())):
            identifier = xi[i]
            if random.random() < self.pm:
                node_neighbors = [m for m in self.graph.neighbors(i)]
                for j in range(len(node_neighbors)):
                    xi[node_neighbors[j]] = identifier
        return xi

if __name__ == "__main__":
    graph = nx.karate_club_graph()
    test = MODEA(10,0.7,1.494,1.494,graph,100,10,0.1,5)
    test.cycling()
    print()

# %%
