#%% Node begins from 0
import networkx as nx
import random
import math
import copy
from itertools import chain,product,combinations
from operator import itemgetter
from copy import deepcopy

                    
class Individual():

    def __init__(self,prop_round,graph,index):
        self.graph = deepcopy(graph)
        self.n = len(self.graph.nodes)
        self.neighbors = []
        self.weight = []
        self.index = index

        unique_list = [i for i in range(0, self.n)]
        random.shuffle(unique_list)
        for i,node in enumerate(self.graph.nodes):
            self.graph.nodes[node]['position'] = unique_list[i]
            self.graph.nodes[node]['velocity'] = random.randint(0,1)
            self.graph.nodes[node]['Pbest'] = self.graph.nodes[node]['position']
            
        self.label_propagation(prop_round)

    def label_propagation(self,rounds):
        nodes = [n for n in self.graph.nodes]
        n = self.n
        for j in range(rounds):
            for node in nodes:
                neighbors = [m for m in self.graph.neighbors(node)]
                if len(neighbors) > 1:
                    community = dict()
                    for neighbor in neighbors:
                        key = self.graph.nodes[neighbor]['position']
                        community[key] = community.get(key,0)+1

                    #check if all values are same if so, random choose one
                    if len(set(community.values()))==1:
                        self.graph.nodes[node]['position'] = random.choice(list(community.keys()))
                    else:
                        self.graph.nodes[node]['position'] = max(community.items(), key=itemgetter(1))[0]
                elif len(neighbors) == 1:
                    self.graph.nodes[node]['position'] =self.graph.nodes[neighbors[0]]['position']


    def get_velocity_as_list(self):
        velocity = [self.graph.nodes[node]['velocity'] for node in self.graph.nodes]
        return velocity

    def set_node_velocity_use_list(self,velocity_list):
        for n,v in enumerate(velocity_list):
            self.graph.nodes[n]['velocity'] = v 
        return
    
    def get_position_as_list(self):
        position = [self.graph.nodes[node]['position'] for node in self.graph.nodes]
        return position

    def set_node_position_use_list(self,position_list):
        for n,v in enumerate(position_list):
            self.graph.nodes[n]['position'] = v 
        return

    def get_pbest_as_list(self):
        Pbest = [self.graph.nodes[node]['Pbest'] for node in self.graph.nodes]
        return Pbest

    def set_pbest_use_list(self,pbest_list):
        for n,v in enumerate(pbest_list):
            self.graph.nodes[n]['Pbest'] = v 
        return

    def get_velocity_as_dict(self):
        velocity = {node:self.graph.nodes[node]['velocity'] for node in self.graph.nodes}
        return velocity
    def get_position_as_dict(self):
        position = {node: self.graph.nodes[node]['position'] for node in self.graph.nodes}
        return position
    
    def get_Pbest_as_dict(self):
        Pbest = {node:self.graph.nodes[node]['Pbest'] for node in self.graph.nodes}
        return Pbest

    def set_KKM_RC(self, KKM,RC):
        self.KKM = KKM
        self.RC = RC

    # def update_self(self,new_xi_pos,new_velocity,KKM,RC):

    # def __eq__(self,other):
    #     return self.weight == other.weight
            

class Population():

    def __init__(self,graph,Pop_size=20,pap_round=1,neighbor_size=5):

        self.graph = graph
        self.Pop_size = Pop_size
        self.pap_round = pap_round
        self.neighbor_size = neighbor_size
        self.init_population()
        self.init_evenly_distributed_weight()
        self.init_neighbors()
        self.init_ref_point()
        self.init_KKM_RC()
        

    def init_KKM_RC(self):
        for x in self.population:
            x_pos = x.get_position_as_list()
            x.KKM, x.RC = self.calculate_KKM_RC(x_pos)


    def init_evenly_distributed_weight(self):
        #the weight includes 0, and the range is [0,1], so evenly divided into 100 parts
        temp = self.Pop_size-1 
        for i,p in enumerate(self.population):
            p.weight.append(i/temp) 
            p.weight.append((temp-i)/temp)


    def init_neighbors(self):
        for i,current_ind in enumerate(self.population):
            neighbors = []
            for j,indv in enumerate(self.population):
                # if i == j:
                #     continue
                # else:
                distance = self.distance(current_ind,indv)
                neighbors.append((indv,indv.index,distance))
            sorted_pop = sorted(neighbors,key=lambda nei:nei[2])
            current_ind.neighbors = [n for i,n in enumerate(sorted_pop) if i < self.neighbor_size]

    def init_population(self):
        population = []
        for i in range(self.Pop_size):
            population.append(Individual(self.pap_round,self.graph,i))
        self.population = population
        return

    def distance(self,i,j):
        i_v = i.weight
        j_v = j.weight
        distance = math.sqrt(sum([(i_v[n]-j_v[n])**2 for n in range(len(i_v))]))
        return distance

    def init_ref_point(self):
        KKM_list = []
        RC_list =[]
        for p in self.population:
            xi_pos = p.get_position_as_list()
            KKM,RC = self.calculate_KKM_RC(xi_pos)
            KKM_list.append(KKM)
            RC_list.append(RC)
        self.ref = (min(KKM_list),min(RC_list))

    def update_ref_point(self,new_KKM,new_RC):

        self.ref = (min([self.ref[0],new_KKM]),min([self.ref[1],new_RC]))
        

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
            if pbest_KKM < new_xi_KKM or pbest_RC < new_xi_RC:
                xi.set_pbest_use_list(new_xi_pos)
        else:
            if new_xi_KKM >= pbest_KKM and new_xi_RC >= pbest_RC:
                return
            else:
                if (xi.weight[0]*new_xi_KKM + xi.weight[1]*new_xi_RC) < (xi.weight[0]* pbest_KKM + xi.weight[1]*pbest_RC):
                    xi.set_pbest_use_list(new_xi_pos)
        return
#%% 

# import networkx.algorithms.community as nx_comm  
# from draw_community import community_layout
# import matplotlib.pyplot as plt
# import statistics as st 
# from sklearn.metrics.cluster import normalized_mutual_info_score as cal_NMI
# def cal_modularity(partition):
    
#     values = set(partition.values())
#     group_list =[]
#     # get modular list
#     for v in values:
#         group_list.append({k for k in partition.keys() if partition[k] == v })

#     m = nx_comm.modularity(ind.graph,group_list)

#     print('modularity for individual {} is {}'.format(i,m))
#     return m
    

# def draw_graph(i,ind,partition):
#     pos = community_layout(graph, partition)
#     labels = nx.get_node_attributes(ind.graph,'club')
#     d = dict(ind.graph.degree)

#     node_size = [v * 10 for v in d.values()]

#     plt.figure(figsize=(15,15)) 

#     nx.draw_networkx(ind.graph, pos,
#                     width=0.5,
#                     style = 'dotted',
#                     with_labels = True, node_shape='^',
#                     node_size=node_size, 
#                     font_size=10,
#                     horizontalalignment = 'right',
#                     alpha=0.8, 
#                     node_color=list(partition.values()))

#     nx.draw_networkx_labels(ind.graph,
#                             pos,
#                             labels=labels,
#                             alpha=0.8,
#                             font_size=10,
#                             horizontalalignment = 'left',
#                             verticalalignment='baseline')
#     plt.title('individual {}'.format(i))              
#     plt.show()
#     return partition

# if __name__ == "__main__":

#     graph = nx.karate_club_graph()
#     pap_round = 10
#     P = Population(graph,Pop_size=100,pap_round=pap_round)

#     result = []
    
#     for i,ind in enumerate(P.population):

#         partition = {k:ind.graph.nodes[k]['position'] for k in ind.graph.nodes()}
#         true_partition = {k:ind.graph.nodes[k]['club'] for k in ind.graph.nodes()}
#         # draw_graph(i,ind,partition)

#         #get modularity for the partition
#         m = cal_modularity(partition)
        
#         NMI = cal_NMI(list(partition.values()),list(true_partition.values()))

#         result.append((i,m,partition,NMI))

#     #sort by max modularity
#     max_m_i = max(result,key=lambda x :x[1])[0]
#     max_m = max(result,key=lambda x :x[1])[1]
#     max_m_partition = max(result,key=lambda x :x[1])[2]
#     mean_m = st.mean([r[1] for r in result])
#     draw_graph(max_m_i,P.population[max_m_i],max_m_partition)
#     print('max modularity is individual {}, {}, mean mdoularity is {}  '.format(max_m_i,max_m,mean_m))

#     #sort by max NMI
#     max_NMI_i = max(result,key=lambda x :x[3])[0]
#     max_NMI_partition = max(result,key=lambda x :x[3])[2]
#     max_NMI = max(result,key=lambda x :x[3])[3]
#     mean_NMI = st.mean([r[3] for r in result])
#     draw_graph(max_NMI_i,P.population[max_NMI_i],max_NMI_partition)
#     print('max NMI is individual {}, {}, mean NMI is {}  '.format(max_NMI_i,max_NMI,mean_NMI))


#     print()


# %%
