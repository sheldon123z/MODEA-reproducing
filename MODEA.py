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
from draw_community import community_layout
from adjustText import adjust_text

IND = 0
IDX = 1 
DIST = 2



class MODEA():

    def __init__(self,Gen,inertia_weight,c1,c2,graph,pop_size,pap_round=10,pm=0.1,neighbor_size=5):

        self.P = Population(graph,pop_size,pap_round,neighbor_size)
        # self.draw_current_objective_graph()
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
                child = copy.deepcopy(pop[i])
                self.gbest = random.choice(child.neighbors)
                xi_pos = child.get_position_as_list()
                new_v = self.calculate_new_velocity(self.gbest,child)
                new_xi_pos = self.calculate_new_position(child,xi_pos,new_v)

                if t < self.pm * self.Gen:
                    new_xi_pos = self.turbulence(new_xi_pos)

                KKM_new,RC_new = self.P.calculate_KKM_RC(new_xi_pos)

                self.update_neighbor_solutions(child,new_xi_pos,new_v,KKM_new,RC_new)
                self.P.update_ref_point(KKM_new,RC_new)
                self.P.update_pbest(pop[i])

            self.draw_current_objective_graph(t)
        
    def draw_current_objective_graph(self,t):
        pop = self.P.population

        RC_result = [pop[i].RC for i in range(len(pop))]
        KKM_result = [pop[i].KKM for i in range(len(pop))]

        points = dict()
        for k,r in zip(KKM_result,RC_result):
            points[(k,r)] = points.get((k,r),0)+1
        KKM = [k[0] for k in points.keys()]
        RC = [k[1] for k in points.keys()]
        counter = [k for k in points.values()]

        plt.figure(figsize=(8,8)) 

        plt.scatter(KKM, RC, marker='^', alpha=0.8)
        plt.xlabel("KKM")
        plt.ylabel("RC")
        # plt.ylim(0,5)
        Text =[]
        for i, (i_x, i_y) in enumerate(zip(KKM, RC)):
            Text.append(plt.text(i_x, i_y, '({}, {}): {}'.format(round(i_x,3), round(i_y,3),counter[i])))
        adjust_text(Text)
        plt.title("generation {}".format(t))
        plt.show()
                        

    def update_neighbor_solutions(self,indiv,new_pos,new_v,KKM_new,RC_new):

        for neighbor in indiv.neighbors:
            neibr_pos = neighbor[IND].get_position_as_list()
            KKM_neibr,RC_neibr = self.P.calculate_KKM_RC(neibr_pos)
            
            neibr_ind = neighbor[IND]
            neibr_idx = neighbor[IDX]

            g_new = max([neibr_ind.weight[0] * abs(KKM_new-self.P.ref[0]),neibr_ind.weight[1] * abs(RC_new-self.P.ref[1])])
            g_old = max([neibr_ind.weight[0] * abs(KKM_neibr-self.P.ref[0]),neibr_ind.weight[1] * abs(RC_neibr-self.P.ref[1])])
            if g_new < g_old:
                neibr_ind.set_node_position_use_list(new_pos)
                neibr_ind.set_node_velocity_use_list(new_v)
                neibr_ind.set_KKM_RC(KKM_neibr,RC_neibr)
                self.P.population[neibr_idx] = neibr_ind




    def sigmoid(self, V):
        Y = []
        for x in V:
            if random.random() < 1/(1+math.exp(-x)):
                Y.append(1)
            else:
                Y.append(0)
        return Y

    def XOR(self,l1,l2):

        return [1-int(bool(i)^bool(j)) for i,j in zip(l1,l2)]

    def calculate_new_velocity(self,gbest,xi):
        gbest_pos = gbest[IND].get_position_as_list()
        xi_pos = xi.get_position_as_list()
        xi_pbest = xi.get_pbest_as_list()
        xi_v = xi.get_velocity_as_list()

        new_v = self.sigmoid(random.random() * np.asfarray(xi_v) + self.c1 * random.random() * np.asfarray(self.XOR(xi_pbest,xi_pos)) + \
         self.c2*np.asfarray(random.random()) * np.asfarray(self.XOR(gbest_pos,xi_pos)))

        return new_v

    def get_nbest(self,node,xi):
        neighbors = [m for m in xi.graph.neighbors(node)]
        if len(neighbors) > 1:
            community = dict()
            for neighbor in neighbors:
                key = xi.graph.nodes[neighbor]['position']
                community[key] = community.get(key,0)+1
            # 从最大的neighbor中随机选择一个
            max_val = max(community.values())
            # 拥有max value 的keys，也就是最多邻居所属的社团标签组成的list
            max_keys = [k for k in community if community[k] == max_val]
            #随机选取一个
            result = random.choice(max_keys)
            # result = max(community.items(), key=itemgetter(1))[0]  
        elif len(neighbors)==1:
            result = xi.graph.nodes[node]['position']
        return result

    def calculate_new_position(self,xi,xi_pos,new_v):

        xi_v = new_v
        new_xi_pos = []
        for node,(vi,p) in enumerate(zip(new_v,xi_pos)):
            if vi == 0:
                new_xi_pos.append(p)
            elif vi == 1:
                Nbest = self.get_nbest(node,xi)
                new_xi_pos.append(Nbest)
        return new_xi_pos

    def turbulence(self,new_pos):
        xi = new_pos
        for i in range(len(self.graph.nodes())):
            identifier = xi[i]
            node_neighbors = [m for m in self.graph.neighbors(i)]
            if random.random() < self.pm:
                for j in range(len(node_neighbors)):
                    xi[node_neighbors[j]] = identifier
        return xi
#%%
import networkx.algorithms.community as nx_comm
import statistics as st
from sklearn.metrics.cluster import normalized_mutual_info_score as cal_NMI
def draw_graph(i,ind,partition,label = 'club', **kwargs):
    pos = community_layout(graph, partition)
    labels = nx.get_node_attributes(ind.graph,label)
    d = dict(ind.graph.degree)

    node_size = [v * 10 for v in d.values()]

    plt.figure(figsize=(15,15)) 

    nx.draw_networkx(ind.graph, pos,
                    width=0.5,
                    style = 'dotted',
                    with_labels = True, node_shape='^',
                    node_size=node_size, 
                    font_size=10,
                    horizontalalignment = 'right',
                    alpha=0.8, 
                    node_color=list(partition.values()),
                    **kwargs)

    nx.draw_networkx_labels(ind.graph,
                            pos,
                            labels=labels,
                            alpha=0.8,
                            font_size=10,
                            horizontalalignment = 'left',
                            verticalalignment='baseline',
                            **kwargs)
    plt.title('individual {}'.format(i))              
    plt.show()
    return partition

def cal_modularity(partition):

    values = set(partition.values())
    group_list =[]
    # get modular list
    for v in values:
        group_list.append({k for k in partition.keys() if partition[k] == v })
    m = nx_comm.modularity(ind.graph,group_list)

    print('modularity for individual {} is {}'.format(i,m))
    return m

if __name__ == "__main__":
    graph = nx.karate_club_graph()

    Gen = 10
    inertia_weight = 0.7
    c1 = c2 = 1.494
    pop_size = 100
    pap_round = 10
    pm = 0.1
    neighbor_size = 30
    model = MODEA(Gen,inertia_weight,c1,c2,graph,pop_size,pap_round,pm,neighbor_size)
    model.cycling()

    result = []
    for i,ind in enumerate(model.P.population):
        partition = {k:ind.graph.nodes[k]['position'] for k in ind.graph.nodes()}
        true_partition = {k:ind.graph.nodes[k]['club'] for k in ind.graph.nodes()}

        #get modularity for the partition
        m = cal_modularity(partition)
        NMI = cal_NMI(list(partition.values()),list(true_partition.values()))
        result.append((i,m,partition,NMI))


    #sort by max modularity
    max_m_i = max(result,key=lambda x :x[1])[0]
    max_m = max(result,key=lambda x :x[1])[1]
    max_m_partition = max(result,key=lambda x :x[1])[2]
    mean_m = st.mean([r[1] for r in result])
    draw_graph(max_m_i,model.P.population[max_m_i],max_m_partition)
    print('max modularity is individual {}, {}, mean mdoularity is {}  '.format(max_m_i,max_m,mean_m))

    #sort by max NMI
    max_NMI_i = max(result,key=lambda x :x[3])[0]
    max_NMI_partition = max(result,key=lambda x :x[3])[2]
    max_NMI = max(result,key=lambda x :x[3])[3]
    mean_NMI = st.mean([r[3] for r in result])
    draw_graph(max_NMI_i,model.P.population[max_NMI_i],max_NMI_partition)
    print('max NMI is individual {}, {}, mean NMI is {}  '.format(max_NMI_i,max_NMI,mean_NMI))
    
    print()



# %%
