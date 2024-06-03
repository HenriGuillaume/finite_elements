import numpy as np
from matplotlib import pyplot as plt
import itertools
from scipy.spatial import Delaunay
import sys

sys.setrecursionlimit(2000)

# nodes are stored as tuples (they need to be hashable)
# meshes are stored as a dictionary where keys are vertices (nodes) and values are a list of their neighbours
# preferably, edges shouldn't intersect (so that our test functions stay 
# somewhat orthogonal) 

def merge_meshes(mesh1, mesh2):
    '''
    merges two meshes
    '''
    merged = mesh1.copy()
    for k in mesh2.keys():
        neighbours = list(set(merged.get(k, []) + mesh2[k]))
        merged[k] = neighbours
    return merged

def merge_recursive(mesh_list):
    '''
    merges a list of meshes recursively
    '''
    if len(mesh_list) == 1:
        return mesh_list[0]
    if len(mesh_list) == 1:
        return merge_meshes(mesh_list[0], mesh_list[1])
    mid = len(mesh_list) // 2
    return merge_meshes(merge_recursive(mesh_list[:mid]), merge_recursive(mesh_list[mid:]))

def simplex(nodes):
    '''returns a simplex, i.e. a mesh where every 
    node is connected to every other one'''
    nodes = [tuple(n) for n in nodes]
    mesh = dict()
    for n in nodes:
        neighbours = []
        for k in nodes:
            if k != n:
                neighbours.append(k)
        mesh[n] = neighbours
    return mesh

def general_mesh(nodes):
    simplices = []
    tri = Delaunay(nodes)
    for s in tri.simplices:
        simplex_nodes = [nodes[i] for i in s]
        new_simplex = simplex(simplex_nodes)
        simplices.append(new_simplex)
    mesh = merge_recursive(simplices)
    return mesh

def circle_nodes(radius, N):
    '''
    generates a circular mesh with simplices with given center and radius
    TO CHECK: SOME NODES DISAPPEAR WHEN USING general_mesh() ????
    '''
    nodes = [(0,0)]
    V = dict() # dicitonnary for potential and boundary conditions
    for i in range(1, N+1):
        r = i * (radius/N)
        angle_div = N * (2**(i-1))
        for j in range(angle_div + 1):
            angle = j * (2 * np.pi) / angle_div
            #p = np.array([r * np.cos(angle), r * np.sin(angle)])
            p = (r * np.cos(angle), r * np.sin(angle))
            nodes.append(p)
            if i == N:
                V[p] = 'bnd'
    return nodes, V


def pyramid_nodes(base, height, N):
    h = height / (N-1)
    nodes = []
    V = dict()
    base_sides = [[i * (l / N) for i in range(N+1)] for l in base] + [[0]]
    nodes = list(itertools.product(*base_sides))
    for n in nodes:
        V[tuple(n)] = 'bnd'
    for i in range(1, N-1):
        # number of points is N-i
        new_sides = [[j * ((l - (i*l)/(N-1)) / (N - 1 - i)) + (i*l) / (2*(N-1)) for j in range(N-i)] for l in base] + [[i*h]]
        new_nodes = list(itertools.product(*new_sides))
        nodes += new_nodes
        for p in new_sides[0]:
            V[(p, new_sides[0][0], i*h)] = 'bnd'
            V[(p, new_sides[0][-1], i*h)] = 'bnd'
        for p in new_sides[1]:
            V[(new_sides[1][0], p, i*h)] = 'bnd'
            V[(new_sides[1][-1], p, i*h)] = 'bnd'
    nodes.append([base[0]/2, base[1]/2, height])
    V[(base[0]/2, base[1]/2, height)] = 'bnd'
    return nodes, V

def cube_nodes(*lengths, N):
    '''
    generates a cubical mesh with simplices given a list of lengths
    '''
    sides = [[i * (l / N) for i in range(N+1)] for l in lengths]
    nodes = list(itertools.product(*sides))
    V = dict()
    if len(lengths) == 2:
        for n in nodes:
            if n[0] == 0 or n[1] == 0 or n[0] == lengths[0] or n[1] == lengths[1]: # check edge nodes
                V[n] = 'bnd'
    if len(lengths) == 3:
        for n in nodes:
            if n[0] == 0 or n[1] == 0 or n[2] == 0 or n[0] == lengths[0] or n[1] == lengths[1] or n[2] == lengths[2]:
                V[n] = 'bnd'
    return nodes, V

def display_2d_mesh(mesh):
    fig = plt.figure()
    ax = fig.add_subplot()
    for node in mesh.keys():
        for neighbour in mesh[node]:
            x = (node[0], neighbour[0])
            y = (node[1], neighbour[1])
            ax.plot(x, y, 'b-')
    plt.show()

def display_3d_mesh(mesh):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for node in mesh.keys():
        for neighbour in mesh[node]:
            x = (node[0], neighbour[0])
            y = (node[1], neighbour[1])
            z = (node[2], neighbour[2])
            ax.plot(x, y, z, 'b-')
    plt.show()
