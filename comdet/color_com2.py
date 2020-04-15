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
    
def color_com_2(A,e,tag_name):
    node_num=[]
    for i in range(len(e)):
        node_num.append(tag_name[i])

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
        elif nod_inf[1][i] < 9: 
            color_map.append('darksalmon')
        elif nod_inf[1][i] < 10: 
            color_map.append('springgreen')
        elif nod_inf[1][i] < 11: 
            color_map.append('aqua')
        elif nod_inf[1][i] < 12: 
            color_map.append('peru')
        elif nod_inf[1][i] < 13: 
            color_map.append('slategray')
        elif nod_inf[1][i] < 14: 
            color_map.append('c')
        elif nod_inf[1][i] < 15: 
            color_map.append('tan')
        elif nod_inf[1][i] < 16: 
            color_map.append('ivory')
        elif nod_inf[1][i] < 17: 
            color_map.append('chocolate')
        elif nod_inf[1][i] < 18: 
            color_map.append('lavender')
        elif nod_inf[1][i] < 19: 
            color_map.append('purple')
        else: color_map.append('green') 
    nx.draw(G,node_color = color_map,with_labels = True) 
    plt.show()
    
def retag(tag,e):
    n=len(tag)
    tagg1=[]
    tagg2=[]
    for i in range(n):
        if e[i]==0:
            tagg1.append(tag[i])
        if e[i]==1:
            tagg2.append(tag[i])
    #print(tagg1)
    #print(tagg2)
    return tagg1,tagg2