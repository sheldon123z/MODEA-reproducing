#%% Node begins from 0
import networkx as nx
import random
import math
import copy
from itertools import chain,product,combinations
from operator import itemgetter
from copy import deepcopy

                    
class Individual():

    def __init__(self,prop_round,graph):
        self.P = []
        self.graph = deepcopy(graph)
        self.nodes = self.graph.nodes()
        self.n = len(self.nodes)
        self.neighbors = []
        self.weight = []

        unique_list = [i for i in range(0, self.n)]
        random.shuffle(unique_list)
        for i,node in enumerate(self.nodes):
            self.nodes[node]['position'] = unique_list[i]
            self.nodes[node]['velocity'] = random.randint(0,1)
            self.nodes[node]['Pbest'] = self.nodes[node]['position']
            
        self.label_propagation(prop_round)

    def label_propagation(self,rounds):
        nodes = [n for n in self.nodes]
        n = self.n
        for j in range(rounds):
            for node in nodes:
                neighbors = [m for m in self.graph.neighbors(node)]
                if len(neighbors) > 0:
                    community = dict()
                    for neighbor in neighbors:
                        key = self.graph.nodes[neighbor]['position']
                        community[key] = community.get(key,0)+1
                    self.graph.nodes[node]['position'] = max(community.items(), key=itemgetter(1))[0]


    def get_velocity_as_list(self):
        velocity = [self.nodes[node]['velocity'] for node in self.nodes]
        return velocity

    def set_node_velocity_use_list(self,velocity_list):
        for n,v in enumerate(velocity_list):
            self.graph.nodes[n]['velocity'] = v 
        return
    
    def get_position_as_list(self):
        position = [self.nodes[node]['position'] for node in self.nodes]
        return position

    def set_node_position_use_list(self,position_list):
        for n,v in enumerate(position_list):
            self.graph.nodes[n]['position'] = v 
        return

    def get_pbest_as_list(self):
        Pbest = [self.nodes[node]['Pbest'] for node in self.nodes]
        return Pbest

    def set_pbest_use_list(self,pbest_list):
        for n,v in enumerate(pbest_list):
            self.graph.nodes[n]['Pbest'] = v 
        return

    def get_velocity_as_dict(self):
        velocity = {node:self.nodes[node]['velocity'] for node in self.nodes}
        return velocity
    def get_position_as_dict(self):
        position = {node: self.nodes[node]['position'] for node in self.nodes}
        return position
    
    def get_Pbest_as_dict(self):
        Pbest = {node:self.nodes[node]['Pbest'] for node in self.nodes}
        return Pbest

    def set_KKM_RC(self, KKM,RC):
        self.KKM = KKM
        self.RC = RC

    # def update_self(self,new_xi_pos,new_velocity,KKM,RC):

    # def __eq__(self,other):
    #     return self.weight == other.weight
            

class Population():

    def __init__(self,graph,Pop_size=20,pap_round=10,neighbor_size=5):

        self.graph = graph
        self.Pop_size = Pop_size
        self.pap_round = pap_round
        self.neighbor_size = neighbor_size
        self.init_population()
        self.init_evenly_distributed_weight()
        self.init_neighbors()
        self.set_ref_point()
        self.init_KKM_RC()
        

    def init_KKM_RC(self):
        for x in self.population:
            x_pos = x.get_position_as_list()
            x.KKM, x.RC = self.calculate_KKM_RC(x_pos)


    def init_evenly_distributed_weight(self):
        temp = self.Pop_size-1 #the weight includes 0, and the range is [0,1], so evenly divided into 100 parts
        for i,p in enumerate(self.population):
            p.weight.append(i/temp) 
            p.weight.append((temp-i)/temp)


    def init_neighbors(self):
        for i,_self in enumerate(self.population):
            neighbors = []
            for j in range(len(self.population)):
                if i == j:
                    continue
                else:
                    distance = self.distance(_self,self.population[j])
                    neighbors.append((self.population[j],i,distance))
            neighbors = sorted(neighbors,key=lambda x:x[1])
            _self.neighbors = [n for i,n in enumerate(neighbors) if i < self.neighbor_size]

    def init_population(self):
        population = []
        for i in range(self.Pop_size):
            population.append(Individual(self.pap_round,self.graph))
        self.population = population
        return

    def distance(self,i,j):
        i_v = i.weight
        j_v = j.weight
        distance = math.sqrt(sum([(i_v[n]-j_v[n])**2 for n in range(len(i_v))]))
        return distance
       
    def set_ref_point(self):
        KKM_list = []
        RC_list =[]
        for p in self.population:
            xi_pos = p.get_position_as_list()
            KKM,RC = self.calculate_KKM_RC(xi_pos)
            KKM_list.append(KKM)
            RC_list.append(RC)
        self.ref = (min(KKM_list),min(RC_list))
        

    def calculate_KKM_RC(self,xi_pos):
        
        cover = []
        community = {}
        for i,n in enumerate(xi_pos):
            community[n] = community.get(n,[])+[i]
        for item in community.items():
            cover.append(item[1])

        k = len(cover)
        n = len(self.graph.nodes())
        A = nx.convert_matrix.to_numpy_array(self.graph)
        
        Ls =[]
        for i in range(k):
            L = sum([A[v][w] for v,w in combinations(cover[i],2)])
            Ls.append(L/len(cover[i]))
        
        KKM = 2*(n-k)-sum(Ls)
        Ls = []
        for i in range(k):
            C_j = copy.deepcopy(cover)
            C_j.remove(cover[i])
            C_j = set(chain.from_iterable(C_j))
            L = sum([A[v][w] for v,w in product(cover[i],C_j)])
            Ls.append(L/len(cover[i]))
        RC = sum(Ls)
        return KKM, RC    

    def update_pbest(self,xi):
        
        xi_pbest = xi.get_pbest_as_list()
        new_xi_pos = xi.get_position_as_list()

        pbest_KKM,pbest_RC = self.calculate_KKM_RC(xi_pbest)
        new_xi_KKM,new_xi_RC = self.calculate_KKM_RC(new_xi_pos)

        if pbest_KKM <= new_xi_KKM and pbest_RC <= new_xi_RC:
            xi.set_pbest_use_list(new_xi_pos)
        else:
            if new_xi_KKM >= pbest_KKM and new_xi_RC >= pbest_RC:
                return
            else:
                if xi.weight[0]*new_xi_KKM + xi.weight[1]*new_xi_RC < xi.weight[0]* pbest_KKM + xi.weight[1]*pbest_RC:
                    xi.set_pbest_use_list(new_xi_pos)
        return
            

if __name__ == "__main__":
    # graph = nx.Graph()
    # graph.add_nodes_from([1,2,3,4,5,6,7])
    # graph.add_edges_from([(3,6),(6,1),(1,2),(1,4),(2,5),(5,7),(2,7)])
    graph = nx.karate_club_graph()

    P = Population(graph)

    print()



# %%
