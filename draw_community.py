# %%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def community_layout(g, partition,intra_community_dist = 30,scale = 3):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=scale)

    pos_nodes = _position_nodes(g, partition, scale=1.,k=intra_community_dist)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node]*4 + pos_nodes[node]*1.5

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

if __name__ == "__main__":

    from community import community_louvain

    g = nx.karate_club_graph()
    partition = community_louvain.best_partition(g)
    pos = community_layout(g, partition)

    labels = nx.get_node_attributes(g,'club')
    # label = nx.draw_networkx_labels(g,pos,labels=labels,alpha=0.5,font_size=5)

    d = dict(g.degree)
    node_size = [v*30 for v in d.values()]
    plt.figure(figsize=(15,15)) 
    nx.draw_networkx(g, pos, width=0.5,style = 'dotted',with_labels = True, node_shape='^', node_size=node_size, font_size=10,horizontalalignment = 'right',alpha=0.8, node_color=list(partition.values()))
    # nx.draw_networkx_edges(g,pos,alpha = 0.5)
    nx.draw_networkx_labels(g,pos,labels=labels,alpha=0.8,font_size=10,horizontalalignment = 'left',verticalalignment='baseline')

    plt.show()

# %%
