import meshes
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d as a3
from scipy.linalg import eigh
import matplotlib.tri as tri

def test_function_2d(x, y, p1, p2, p3):
    '''
    test function with value 1 at point p1
    '''
    a = (p2[1] - p3[1])*(x - p3[0]) - (p2[0] - p3[0])*(y - p3[1])
    b = (p1[0] - p3[0])*(p2[1] - p3[1]) - (p1[1] - p3[1])*(p2[0] - p3[0])
    return a/b


def common_neighbours(mesh, n1, n2):
    common_list = []
    for neighbour in mesh[n1]:
        if neighbour in mesh[n2]:
            common_list.append(neighbour)
    return common_list

def common_to_three(mesh, n1, n2, n3):
    '''
    yes, this function could've been generalised but this code isn't meant to be generalised
    '''
    common_list = []
    for neighbour in mesh[n1]:
        if neighbour in mesh[n2] and neighbour in mesh[n3]:
            common_list.append(neighbour)
    return common_list

def monomial_integrals_2d(pi, pj, p3):
    '''
    compute the integral of the necessary "shifted" monomials over the canonical simplex
    '''
    pix, piy = pi[0], pi[1]
    pjx, pjy = pj[0], pj[1]
    p3x, p3y = p3[0], p3[1]
    # first we need the coefficients of the transformation matrix
    T = np.array([[p3x-pix, pjx-pix], [p3y-piy, pjy-piy]])
    a, b, c, d = T[0][0], T[0][1], T[1][0], T[1][1]
    # integral for x_1*X_2
    IX1X2 = (a*c+b*d)/12 + (a*d+b*c)/24 + (pix*c + piy*a)/6 + (piy*b + pix*d)/6 + pix*piy/2
    # integral for x_1*X_2
    IX1sqrd = (a*a + b*b)/12 + 2*a*b/24 + (2*a*pix + 2*b*pix)/6 + pix*pix/2
    # integral for x_1*X_2
    IX2sqrd = (c*c + d*d)/12 + 2*c*d/24 + (2*c*piy + 2*d*piy)/6 + piy*piy/2
    # integral for x_1*X_2
    IX1 = (a+b)/6 + pix/2
    # integral for x_1*X_2
    IX2 = (c+d)/6 + piy/2
    # integral for x_1*X_2
    I1 = 1/2
    # determinant for the substitution
    det = abs(np.linalg.det(T))
    return {'IX1X2': IX1X2, 'IX1sqrd': IX1sqrd, 'IX2sqrd': IX2sqrd, 'IX1': IX1, 'IX2': IX2, 'I1': I1, 'det':det}


def mass_coeff_2d(pi, pj, p3):
    '''
    Computes (part of) the coefficient of the mass matrix over a SIMPLEX
    '''
    pix, piy = pi[0], pi[1]
    pjx, pjy = pj[0], pj[1]
    p3x, p3y = p3[0], p3[1]
    # denominator constant in function
    denominator = ((pix-p3x)*(pjy-p3y) - (piy-p3y)*(pjx-p3x)) * ((pjx-p3x)*(piy-p3y) - (pjy-p3y)*(pix-p3x))
    # we write the polynomial part as AX_1^2 + BX_2^2 + CX_1X_2 + DX_1 + EX_2 + F 
    #X_1^2
    A = (pjy-p3y)*(piy-p3y)
    #X_2^2
    B = (pjx-p3x)*(pix-p3x)
    #X_1*X_2
    C = (-1)*((pjy-p3y)*(pix-p3x) + (pjx-p3x)*(piy-p3y))
    #X_1
    D = (pjy-p3y) * ((pix-p3x)*p3y - (piy-p3y)*p3x) + (piy-p3y) * ((pjx-p3x)*p3y - (pjy-p3y)*p3x)
    #X_2
    E = (-1)*((pjx-p3x) * ((pix-p3x)*p3y - (piy-p3y)*p3x) + (pix-p3x) * ((pjx-p3x)*p3y - (pjy-p3y)*p3x))
    #constant
    F = (pjy-p3y)*(piy-p3y)*p3x*p3x - (pjy-p3y)*(pix-p3x)*p3y*p3x - (piy-p3y)*(pjx-p3x)*p3x*p3y + (pjx-p3x)*(pix-p3x)*p3y*p3y
    # now we get the integral of each monomial
    I = monomial_integrals_2d(pi, pj, p3)
    S = A*I['IX1sqrd'] + B*I['IX2sqrd'] + C*I['IX1X2'] + D*I['IX1'] + E*I['IX2'] + F*I['I1']
    return (-1)*(I['det']/denominator) * S


def mass_diag_coeff_2d(pi, pj, p3):
    '''
    compute the diagonal mass coefficients, i.e. the integral of the squared test function over a simplex
    I'm aware that the code is redundant, but it's not that big a deal
    '''
    pix, piy = pi[0], pi[1]
    pjx, pjy = pj[0], pj[1]
    p3x, p3y = p3[0], p3[1]
    denominator = ((pix-p3x)*(pjy-p3y) - (piy-p3y)*(pjx-p3x))**2
    K = (-2)*(pjy-p3y)*(pjx-p3x)
    A = (pjy-p3y)**2
    B = (pjx-p3x)**2
    C = K
    D = (-2)*p3x*(pjy-p3y)*(pjy-p3y) - p3y*K
    E = (-2)*p3y*(pjx-p3x)*(pjx-p3x) - p3x*K
    F = p3x*p3x*(pjy-p3y)*(pjy-p3y) + p3y*p3y*(pjx-p3x)*(pjx-p3x) + p3x*p3y*K
    # integrals
    I = monomial_integrals(pi, pj, p3)
    S = A*I['IX1sqrd'] + B*I['IX2sqrd'] + C*I['IX1X2'] + D*I['IX1'] + E*I['IX2'] + F*I['I1']
    return (-1)*(I['det']/denominator) * S     
  

def stiffness_coeff_2d(pi, pj, p3):
    '''
    Computes (part of) the coefficient of the stiffness matrix over a SIMPLEX
    '''
    pix, piy = pi[0], pi[1]
    pjx, pjy = pj[0], pj[1]
    p3x, p3y = p3[0], p3[1]
    C1 = (pix-p3x)*(pjy-p3y) - (piy-p3y)*(pjx-p3x)
    grad1 = (1/C1) * np.array([pjy-p3y, p3x-pjx])
    C2 = (pjx-p3x)*(piy-p3y) - (pjy-p3y)*(pix-p3x)
    grad2 = (1/C2) * np.array([piy-p3y, p3x-pix])
    product = np.dot(grad1, grad2)
    # compute the area of the simplex
    a, b, c, d = p3x-pix, pjx-pix, p3y-piy, pjy-piy
    area = abs(a*d - b*c)/2 
    return product*area


def stiffness_diag_coeff_2d(pi, pj, p3):
    '''
    compute the diagonal stiffness coefficients, i.e. the integral of the squared gradient over a simplex
    '''
    pix, piy = pi[0], pi[1]
    pjx, pjy = pj[0], pj[1]
    p3x, p3y = p3[0], p3[1]
    C1 = (pix-p3x)*(pjy-p3y) - (piy-p3y)*(pjx-p3x)
    grad1 = (1/C1) * np.array([pjy-p3y, p3x-pjx])
    product = np.dot(grad1, grad1)
    # compute the area of the simplex
    a, b, c, d = p3x-pix, pjx-pix, p3y-piy, pjy-piy
    area = abs(a*d - b*c)/2 
    return product*area


def monomial_integrals_3d(pi, pj, p3, p4):
    '''
    compute the integral of the necessary "shifted" monomials over the canonical simplex
    this function is mostly about the variable change, the rest is deduced by linearity
    '''
    pix, piy, piz = pi[0], pi[1], pi[2]
    pjx, pjy, pjz = pj[0], pj[1], pj[2]
    p3x, p3y, p3z = p3[0], p3[1], p3[2]
    p4x, p4y, p4z = p4[0], p4[1], p4[2]
    a, b, c, d, e, f, g, h, i = pjx-pix, p3x-pix, p4x-pix, pjy-piy, p3y-piy, p4y-piy, pjz-piz, p3z-piz, p4z-piz
    T = np.array([[a,b,c], [d,e,f], [g,h,i]])
    IXsqrd = (a*a + b*b + c*c + a*b + a*c + b*c)/60 + (a*pix + b*pix + c*pix)/12 + pix*pix/6
    IYsqrd = (d*d + e*e + f*f + d*e + d*f + e*f)/60 + (d*piy + e*piy + f*piy)/12 + piy*piy/6
    IZsqrd = (g*g + h*h + i*i + g*h + g*i + h*i)/60 + (g*piz + h*piz + i*piz)/12 + piz*piz/6
    IXY = (a*d + b*e + c*f)/60 + (a*(e+f) + b*(d+f) + c*(d+e))/120 + (piy*(a+b+c) + pix*(d+e+f))/24 + pix*piy/6
    IXZ = (a*g + b*h + c*i)/60 + (a*(h+i) + b*(g+i) + c*(g+h))/120 + (piz*(a+b+c) + pix*(g+h+i))/24 + pix*piz/6
    IYZ = (g*d + h*e + i*f)/60 + (g*(e+f) + h*(d+f) + i*(d+e))/120 + (piy*(g+h+i) + piz*(d+e+f))/24 + piz*piy/6
    IX = (a+b+c)/24 + pix/6
    IY = (d+e+f)/24 + piy/6
    IZ = (g+h+i)/24 + piz/6
    I1 = 1/6
    det = abs(np.linalg.det(T))
    return {'IXsqrd':IXsqrd, 'IYsqrd':IYsqrd, 'IZsqrd':IZsqrd, 'IXY':IXY, 'IXZ':IXZ, 
            'IYZ':IYZ, 'IX':IX, 'IY':IY, 'IZ':IZ, 'I1': I1, 'det':det}


def mass_coeff_3d(pi, pj, p3, p4):
    pix, piy, piz = pi[0], pi[1], pi[2]
    pjx, pjy, pjz = pj[0], pj[1], pj[2]
    p3x, p3y, p3z = p3[0], p3[1], p3[2]
    p4x, p4y, p4z = p4[0], p4[1], p4[2]
    M = np.array([[pix - p4x, piy - p4y, piz - p4z],
        [pjx - p4x, pjy - p4y, pjz - p4z],
        [p3x - p4x, p3y - p4y, p3z - p4z]])
    # the other vertices function matrix is the same but the two frist lines flipped
    N = np.array([[pjx - p4x, pjy - p4y, pjz - p4z],
        [pix - p4x, piy - p4y, piz - p4z],
        [p3x - p4x, p3y - p4y, p3z - p4z]])

    denominator = np.linalg.det(M) * np.linalg.det(N)
    if denominator == 0:
        return 0
    d1p1, d1p2 = np.linalg.det(M[np.ix_([1,2], [1,2])]), np.linalg.det(N[np.ix_([1,2], [1,2])])
    d2p1, d2p2 = np.linalg.det(M[np.ix_([1,2], [0,2])]), np.linalg.det(N[np.ix_([1,2], [0,2])])
    d3p1, d3p2 = np.linalg.det(M[np.ix_([1,2], [0,1])]), np.linalg.det(N[np.ix_([1,2], [0,1])])
    # we write the polynomial part as AX^2 + BY^2 + CZ^2 + D*XY + E*XZ + F*YZ + G*X + H*Y + I*Z +J
    # first, some useful constants
    C1 = (d1p1*d2p2 + d1p2*d2p1)
    C2 = (d1p1*d3p2 + d1p2*d3p1)
    C3 = (d2p1*d3p2 + d2p2*d3p1)
    A = d1p1*d1p2; B = d2p1*d2p2; C = d3p1*d3p2
    D = (-1)*C1; E = C2; F = (-1)*C3
    G = -2*p4x*d1p1*d1p2 + p4y*C1 - p4z*C2
    H = -2*p4y*d2p1*d2p2 + p4x*C1 + p4z*C3
    I = -2*p4z*d3p1*d3p2 - p4x*C2 + p4y*C3
    J = p4x*p4x * d1p1*d1p2 + p4y*p4y * d2p1*d2p2 + p4z*p4z * d3p1*d3p2 - p4x*p4y*C1 + p4x*p4z*C2 - p4y*p4z*C3
    i = monomial_integrals_3d(pi, pj, p3, p4)
    S = A*i['IXsqrd'] + B*i['IYsqrd'] + C*i['IZsqrd'] + D*i['IXY'] + E*i['IXZ'] + F*i['IYZ'] + G*i['IX'] + H*i['IY'] + I*i['IZ'] + J*i['I1']
    return (i['det']/denominator) * S     


def mass_diag_coeff_3d(pi, pj, p3, p4):
    pix, piy, piz = pi[0], pi[1], pi[2]
    pjx, pjy, pjz = pj[0], pj[1], pj[2]
    p3x, p3y, p3z = p3[0], p3[1], p3[2]
    p4x, p4y, p4z = p4[0], p4[1], p4[2]
    M = np.array([[pix - p4x, piy - p4y, piz - p4z],
        [pjx - p4x, pjy - p4y, pjz - p4z],
        [p3x - p4x, p3y - p4y, p3z - p4z]])
    denominator = np.linalg.det(M) ** 2
    if denominator == 0:
        return 0
    d1p1 = np.linalg.det(M[np.ix_([1,2], [1,2])])
    d2p1 = np.linalg.det(M[np.ix_([1,2], [0,2])])
    d3p1 = np.linalg.det(M[np.ix_([1,2], [0,1])])
    # we write the polynomial part as AX^2 + BY^2 + CZ^2 + D*XY + E*XZ + F*YZ + G*X + H*Y + I*Z +J
    A = d1p1**2; B = d2p1**2; C = d3p1**2
    D = -2*d1p1*d2p1; E = 2*d1p1*d3p1; F = -2*d2p1*d3p1
    G = -2*p4x*d1p1*d1p1 + 2*p4y*d1p1*d2p1 -2*p4z*d1p1*d3p1 
    H = -2*p4y*d2p1*d2p1 +2*p4x*d1p1*d2p1 + 2*p4z*d2p1*d3p1
    I = -2*p4z*d3p1*d3p1 -2*p4x*d1p1*d3p1 + 2*p4y*d2p1*d3p1
    J = (p4x*d1p1)**2 + (p4y*d2p1)**2 + (p4z*d3p1)**2 + 2*((-1)*p4x*p4y*d1p1*d2p1 + p4x*p4z*d1p1*d3p1 + (-1)*p4y*p4z*d2p1*d3p1) 
    i = monomial_integrals_3d(pi, pj, p3, p4)
    S = A*i['IXsqrd'] + B*i['IYsqrd'] + C*i['IZsqrd'] + D*i['IXY'] + E*i['IXZ'] + F*i['IYZ'] + G*i['IX'] + H*i['IY'] + I*i['IZ'] + J*i['I1']
    return (i['det']/denominator) *S


def stiffness_coeff_3d(pi, pj, p3, p4):
    pix, piy, piz = pi[0], pi[1], pi[2]
    pjx, pjy, pjz = pj[0], pj[1], pj[2]
    p3x, p3y, p3z = p3[0], p3[1], p3[2]
    p4x, p4y, p4z = p4[0], p4[1], p4[2]
    M = np.array([[pix - p4x, piy - p4y, piz - p4z],
        [pjx - p4x, pjy - p4y, pjz - p4z],
        [p3x - p4x, p3y - p4y, p3z - p4z]])
    # other point
    N = np.array([[pjx - p4x, pjy - p4y, pjz - p4z],
        [pix - p4x, piy - p4y, piz - p4z],
        [p3x - p4x, p3y - p4y, p3z - p4z]])

    denominator = (np.linalg.det(M) * np.linalg.det(N))
    if denominator == 0:
        return 0
    d1p1, d1p2 = np.linalg.det(M[np.ix_([1,2], [1,2])]), np.linalg.det(N[np.ix_([1,2], [1,2])])
    d2p1, d2p2 = np.linalg.det(M[np.ix_([1,2], [0,2])]), np.linalg.det(N[np.ix_([1,2], [0,2])])
    d3p1, d3p2 = np.linalg.det(M[np.ix_([1,2], [0,1])]), np.linalg.det(N[np.ix_([1,2], [0,1])])
    prod = np.dot(np.array([d1p1, (-1)*d2p1, d3p1]), np.array([d1p2, (-1)*d2p2, d3p2]))
    volume = abs(np.linalg.det(M)) / 2
    return (1/denominator) * prod * volume


def stiffness_diag_coeff_3d(pi, pj, p3, p4):
    pix, piy, piz = pi[0], pi[1], pi[2]
    pjx, pjy, pjz = pj[0], pj[1], pj[2]
    p3x, p3y, p3z = p3[0], p3[1], p3[2]
    p4x, p4y, p4z = p4[0], p4[1], p4[2]
    M = np.array([[pix - p4x, piy - p4y, piz - p4z],
        [pjx - p4x, pjy - p4y, pjz - p4z],
        [p3x - p4x, p3y - p4y, p3z - p4z]])
    
    denominator = np.linalg.det(M)**2
    if denominator == 0:
        #print(np.linalg.det([np.array(pi)-np.array(p4), np.array(pj)-np.array(p4), np.array(p3)-np.array(p4)]))
        # the above is necessarliy 0
        return 0
    d1p1 = np.linalg.det(M[np.ix_([1,2], [1,2])])
    d2p1 = np.linalg.det(M[np.ix_([1,2], [0,2])])
    d3p1 = np.linalg.det(M[np.ix_([1,2], [0,1])])
    prod = np.dot(np.array([d1p1, (-1)*d2p1, d3p1]), np.array([d1p1, (-1)*d2p1, d3p1]))
    volume = abs(np.linalg.det(M)) / 2
    return (1/denominator) * prod * volume
    

def browse_mesh_3d(mesh):
    '''
    browse the mesh and create the stiffness and mass matrices
    '''
    node_list = list(mesh.keys())
    n = len(node_list)
    K = np.zeros((n,n)) # stiffness matrix
    M = np.zeros((n,n)) # mass matrix
    for i, node in enumerate(node_list):
        for neighbour in mesh[node]:
            j = node_list.index(neighbour)
            if neighbour == node: # just in case
                print("this shouldn't happen")
            if j <= i: # only build half of each matrix since they're symmetrical
                # find common neighbours
                common_list = common_neighbours(mesh, node, neighbour)
                # for each common neighbour, apply simplex formulae
                for common in common_list:
                    common_to_three_list = common_to_three(mesh, node, neighbour, common)
                    for last_one in common_to_three_list:
                        K[i][j] += stiffness_coeff_3d(node, neighbour, common, last_one)
                        M[i][j] += mass_coeff_3d(node, neighbour, common, last_one)
    # make these bad boys symmetrical
    K = K + np.transpose(K)
    M = M + np.transpose(M)
    # fill the diagonals while CAREFULLY going over each element only once
    for i, node in enumerate(node_list):
        # keep track of the simplices over which we have already done computations
        already_computed = []
        for neighbour in mesh[node]:
            common_list = common_neighbours(mesh, node, neighbour)
            for common in common_list:
                common_to_three_list = common_to_three(mesh, node, neighbour, common)
                for last_one in common_to_three_list:
                    if sorted((neighbour, common, last_one)) not in already_computed:
                        K[i][i] += stiffness_diag_coeff_3d(node, neighbour, common, last_one)
                        M[i][i] += mass_diag_coeff_3d(node, neighbour, common, last_one)
                        already_computed.append(sorted((neighbour, common, last_one)))
    return K, M

def browse_mesh_2d(mesh):
    '''
    browse the mesh and create the stiffness and mass matrices
    '''
    node_list = list(mesh.keys())
    n = len(node_list)
    K = np.zeros((n,n)) # stiffness matrix
    M = np.zeros((n,n)) # mass matrix
    for i, node in enumerate(node_list):
        for neighbour in mesh[node]:
            j = node_list.index(neighbour)
            if neighbour == node: # just in case
                print("this shouldn't happen")
            if j <= i: # only build half of each matrix since they're symmetrical
                # find common neighbours
                common_list = common_neighbours(mesh, node, neighbour)
                # for each common neighbour, apply simplex formulae
                for common in common_list:
                    K[i][j] += stiffness_coeff_2d(node, neighbour, common)
                    M[i][j] += mass_coeff_2d(node, neighbour, common)
    # make these bad boys symmetrical
    K = K + np.transpose(K)
    M = M + np.transpose(M)
    # fill the diagonals while CAREFULLY going over each element only once
    for i, node in enumerate(node_list):
        # keep track of the simplices over which we have already done computations
        already_computed = []
        for neighbour in mesh[node]:
            common_list = common_neighbours(mesh, node, neighbour)
            for common in common_list:
                if sorted((neighbour, common)) not in already_computed:
                    K[i][i] += stiffness_diag_coeff_2d(node, neighbour, common)
                    M[i][i] += mass_diag_coeff_2d(node, neighbour, common)
                    already_computed.append(sorted((neighbour, common)))
    return K, M

def display_vector(mesh, vect):
    node_list = list(mesh.keys())
    X = [node[0] for node in node_list]
    Y = [node[1] for node in node_list]
    Z = np.array([z for z in vect])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i, node in enumerate(node_list):
        for neighbour in mesh[node]:
            x = (node[0], neighbour[0])
            y = (node[1], neighbour[1])
            #z = (vect[i], vect[node_list.index(neighbour)])
            z = (abs(vect[i]), abs(vect[node_list.index(neighbour)]))
            ax.plot(x, y, z, 'b-')
    plt.show()

def display_contour(mesh, vect_list, vals):
    node_list = list(mesh.keys())
    n = len(vect_list)
    fig, axs = plt.subplots(1, n, sharex='col', sharey='row')
    X = [node[0] for node in node_list]
    Y = [node[1] for node in node_list]
    for i, vect in enumerate(vect_list):
        axs[i].set_title(f'$E=${vals[i]}', fontsize = 14)
        x = [node[0] for node in node_list]
        y = [node[1] for node in node_list]
        z = [abs(k)**2 for k in vect]
        ngridx = 200
        ngridy = 200
        # Create grid values first.
        xi = np.linspace(0, 1, ngridx)
        yi = np.linspace(0, 1, ngridy)
        # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)
        axs[i].contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
        cntr1 = axs[i].contourf(xi, yi, zi, levels=14, cmap="RdBu_r")
        fig.colorbar(cntr1, ax=axs[i], label='$|\psi|^2$')
    plt.show()

def display_3d_density(mesh, vect, bnd_nodes=None):
    node_list = list(mesh.keys())
    probabilities = [abs(k)**2 for k in vect]
    big = max(probabilities)
    small = min(probabilities)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    if bnd_nodes != None:
        for node in bnd_nodes:
            for neighbour in mesh[node]:
                if neighbour in bnd_nodes:
                    x = (node[0], neighbour[0])
                    y = (node[1], neighbour[1])
                    z = (node[2], neighbour[2])
                    ax.plot(x, y, z, 'b-', alpha=0.6, lw=0.5)
    for i, node in enumerate(node_list):
        p = probabilities[i]
        # define color gradient function
        t = (p-small)/(big-small)
        c = np.array([t, 0, (1-t)])
        if p > 0:
            ax.plot([node[0]], [node[1]], [node[2]], markerfacecolor=tuple(c), marker='o', markersize=20, alpha=0.6)
    plt.show()


if __name__ == "__main__":
    #nodes, V = meshes.cube_nodes(1,1,1, N=7)
    nodes, V = meshes.pyramid_nodes([1,1], 1, N=10)
    mesh = meshes.general_mesh(nodes)
    node_list = list(mesh.keys())
    bnd_nodes = [n for n in node_list if V.get(n, None) == 'bnd']
    bnd = [node_list.index(n) for n in bnd_nodes]
    #meshes.display_2d_mesh(test)
    matrices = browse_mesh_3d(mesh)
    K = matrices[0]
    M = matrices[1]
    n = np.size(K, 0)
    #---------------#
    # IMPLEMENT BOUNDARY CONDITIONS
    K = np.delete(K, bnd, 0)
    K = np.delete(K, bnd, 1)
    M = np.delete(M, bnd, 0)
    M = np.delete(M, bnd, 1)
    # cholesky decomposition allows us to solve the eigenvalue problem with np.eigh which gives results
    # sorted by magnitude of the eigenvalues, which np.eig doesn't necessarily do
    L = np.linalg.cholesky(M)
    Linv = np.linalg.inv(L)
    P = Linv @ K @ np.transpose(Linv)
    eigenvalues, eigenvectors = np.linalg.eigh(P)
    eigen_sorted = sorted([abs(e) for e in eigenvalues])
    print(f'eigenvalues modules: {eigen_sorted[:8]}')
    exit()
    dof = [i for i in range(n) if i not in bnd]

    solutions = []
    for i in range(len(eigenvectors)):
        # rebuild the vectors from our alternative formulation
        sol = np.transpose(Linv) @ eigenvectors[:, i]
        vec = np.zeros(n)
        vec[dof] = sol / np.linalg.norm(sol)
        solutions.append(vec)
    
    #display_contour(mesh, solutions[0:4], eigenvalues[0:4])
    #display_vector(mesh, solutions[3])
    for i in range(5):
        v = eigen_sorted[i]
        if v in eigenvalues:
            j = list(eigenvalues).index(v)
        else:
            j = list(eigenvalues).index(-1*v)
        display_3d_density(mesh, solutions[j], bnd_nodes=bnd_nodes)
