#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:17:47 2019

@author: kanekoakihiro
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


"""
二色までならこれを使う.
"""

def color_com2(A,e):
    node_num=[]
    for i in range(len(e)):
        node_num.append(i)

    nod_inf=np.stack([node_num,e])
    nodes = nod_inf[0]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    edges = []
    for hi, hv  in enumerate(A):
        for wi, wv in enumerate(hv):
            if(wv): edges.append((nodes[hi], nodes[wi]))
        
    G.add_edges_from(edges)

    color_map = [] 
    for i in range(len(A)): 
        if nod_inf[1][i] <1: 
            color_map.append('red') 
        else: color_map.append('blue') 
    
    nx.draw(G,node_color = color_map,with_labels = True) 
    plt.show()
    
def color_com3(A,e):
    node_num=[]
    for i in range(len(e)):
        node_num.append(i)

    nod_inf=np.stack([node_num,e])
    nodes = nod_inf[0]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    edges = []
    for hi, hv  in enumerate(A):
        for wi, wv in enumerate(hv):
            if(wv): edges.append((nodes[hi], nodes[wi]))
        
    G.add_edges_from(edges)

    color_map = [] 
    for i in range(len(A)): 
        if nod_inf[1][i] < 1: 
            color_map.append('red')
        elif nod_inf[1][i] < 2: 
            color_map.append('blue')
        else: color_map.append('orange') 
    nx.draw(G,node_color = color_map,with_labels = True) 
    plt.show()
    
def color_com(A,e):
    node_num=[]
    for i in range(len(e)):
        node_num.append(i)

    nod_inf=np.stack([node_num,e])
    nodes = nod_inf[0]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    edges = []
    for hi, hv  in enumerate(A):
        for wi, wv in enumerate(hv):
            if(wv): edges.append((nodes[hi], nodes[wi]))
        
    G.add_edges_from(edges)

    color_map = [] 
    for i in range(len(A)): 
        if nod_inf[1][i] < 1: 
            color_map.append('deepskyblue')
        elif nod_inf[1][i] < 2: 
            color_map.append('pink')
        elif nod_inf[1][i] < 3: 
            color_map.append('orange')
        elif nod_inf[1][i] < 4: 
            color_map.append('blue')
        elif nod_inf[1][i] < 5: 
            color_map.append('red')
        elif nod_inf[1][i] < 6: 
            color_map.append('gold')
        elif nod_inf[1][i] < 7: 
            color_map.append('yellowgreen')
        elif nod_inf[1][i] < 8: 
            color_map.append('purple')
        else: color_map.append('green') 
    nx.draw(G,node_color = color_map,with_labels = True) 
    plt.show()