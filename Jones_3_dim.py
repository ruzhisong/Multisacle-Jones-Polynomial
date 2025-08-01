# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:18:07 2024

@author: Lenovo
"""

import math
import numpy as np
import random
import copy
from scipy.integrate import dblquad
import itertools
from collections import defaultdict
from Bio import PDB


##############################
#gpt

def random_unit_vector():
    # 生成随机的球面坐标
    theta = random.uniform(0, math.pi)
    phi = random.uniform(0, 2 * math.pi)

    # 将球面坐标转换为直角坐标系中的向量
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)

    # 归一化向量
    vec = np.array([x, y, z])
    #vec = np.array([float(11), float(4), float(90)])
    vec /= np.linalg.norm(vec)
    
    return vec

def projection_random_matrix():
    v = random_unit_vector()
    # 构造投影矩阵
    I = np.eye(3)
    Matrix = I - np.outer(v, v)
    
    return Matrix

def project_vectors(coords):
    P = projection_random_matrix()
    # 将每个点投影到以 v 为法向量的平面上
    coords_proj = np.dot(coords, P.T)
    return coords_proj


#############################


#还不理解neigh_array的作用

def generate_neigh_array(n, closed):
    """
    创建一个包含每个原子邻居索引的向量的向量。
    例如，neigh_array[1] 包含原子 1 的邻居索引，即 0 和 2。

    Args:
    - n: 系统中原子的数量
    - closed: 如果系统中最后一个原子连接到第一个原子，则为 True；否则为 False

    Returns:
    - 包含每个原子邻居索引的向量的向量
    """
    neigh_array = []
    
    # 填充原子 1 到 n-2 的邻居
    for i in range(1, n - 1):
        neigh_array.append([i - 1, i + 1])
    
    if closed:
        # 将最后一个原子（索引为 n-1）连接到第一个原子（索引为 0）
        neigh_array.append([n - 2, 0])
        # 将第一个原子（索引为 0）连接到第二个原子（索引为 1）
        neigh_array.insert(0, [n - 1, 1])
    else:
        # 最后一个原子到第一个原子不连接（使用 -1 表示虚构邻居）
        neigh_array.append([n - 2, -1])
        # 将第一个原子（索引为 0）连接到第二个原子（索引为 1）
        neigh_array.insert(0, [-1, 1])
    
    return neigh_array



def dfs(neigh_array, node, visited, parent):
    """
    Helper function to perform DFS and check for loops.

    :param neigh_array: List of lists that contain the indices of an atom's neighbors
    :param node: Current node being visited
    :param visited: List to track visited nodes
    :param parent: Parent node of the current node
    :return: True if a loop is found, otherwise False
    """
    visited[node] = True
    for neighbor in neigh_array[node]:
        if neighbor == -1:
            continue
        if not visited[neighbor]:
            if dfs(neigh_array, neighbor, visited, node):
                return True
        elif neighbor != parent:
            return True
    return False



def count_loops(neigh_array):
    """
    Counts the number of loops formed by order of atoms in neigh_array.
    Used to calculate the bracket polynomial of each combination of crossing annealments.

    :param neigh_array: List of lists that contain the indices of an atom's neighbors
    :param n: The number of atoms in the system
    :return: The number of loops formed by the system
    """
    n = len(neigh_array)
    visited = [False] * n
    loops = 0

    for i in range(n):
        if not visited[i]:
            if dfs(neigh_array, i, visited, -1):
                loops += 1
    #print(loops)
    return loops









def calculate_GL(a1,a2, b1,b2):
    """
    a:tuple; elements are head and tail
    of line segment, each is a (3,) array representing the xyz coordinate.
    """
    # a0, a1 = a[0],a[1]
    # b0, b1 = b[0],b[1]
    a = (np.array(a1), np.array(a2))
    b = (np.array(b1), np.array(b2))
    R = np.empty((2, 2), dtype=tuple)
    for i in range(2):
        for j in range(2):
            R[i, j] = a[i] - b[j]

    n = []
    cprod = []

    cprod.append(np.cross(R[0, 0], R[0, 1]))
    cprod.append(np.cross(R[0, 1], R[1, 1]))
    cprod.append(np.cross(R[1, 1], R[1, 0]))
    cprod.append(np.cross(R[1, 0], R[0, 0]))

    for c in cprod:
        n.append(c / (np.linalg.norm(c) + 1e-6))

    area1 = np.arcsin(np.dot(n[0], n[1]))
    area2 = np.arcsin(np.dot(n[1], n[2]))
    area3 = np.arcsin(np.dot(n[2], n[3]))
    area4 = np.arcsin(np.dot(n[3], n[0]))

    sign = np.sign(np.cross(a[1] - a[0], b[1] - b[0]).dot(a[0] - b[0]))
    Area = area1 + area2 + area3 + area4

    return sign * Area







def simple_mult(a, b):
    """
    Multiply two polynomials where both are passed as parameters.
    
    :param a: The first polynomial represented as a dictionary {power: coefficient}.
    :param b: The second polynomial represented as a dictionary {power: coefficient}.
    :return: The product of a and b represented as a dictionary {power: coefficient}.
    """
    result = {}
    for power_a, coeff_a in a.items():
        for power_b, coeff_b in b.items():
            power = power_a + power_b
            coeff = coeff_a * coeff_b
            if power in result:
                result[power] += coeff
            else:
                result[power] = coeff
    return result



    





#############################
#gpt
def are_collinear_3(p1, p2, p3, p4):  # 3维
    """Check if 4 points are collinear in 3D space."""
    p1, p2, p3, p4 = map(np.array, [p1, p2, p3, p4])
    v1 = p2 - p1
    v2 = p4 - p3

    # 计算叉积
    cross = np.cross(v1, v2)
    
    # 检查叉积的模是否接近于零
    return np.linalg.norm(cross) < 1e-9


def intersect_3(a1, a2, b1, b2):  # 3-维
    """
    Check if two line segments in 3D space intersect and calculate the intersection point if they do.

    Parameters:
    - a1, a2: Endpoints of the first line segment (points as lists or numpy arrays [x, y, z])
    - b1, b2: Endpoints of the second line segment (points as lists or numpy arrays [x, y, z])

    Returns:
    - True and intersection point as a list [x, y, z] if segments intersect
    - False otherwise
    """
    
    a1, a2, b1, b2 = map(np.array, [a1, a2, b1, b2])
    
    # 检查是否共线
    if are_collinear_3(a1, a2, b1, b2):
        return False, None, None, None

    # 计算方向向量
    direction1 = a2 - a1
    direction2 = b2 - b1
    A = np.array([direction1, -direction2]).T
    b = b1 - a1
    
    # 求解线性方程组
    try:
        solution = np.linalg.lstsq(A, b, rcond=None)[0]
        t, s = solution
    except np.linalg.LinAlgError:
        return False, None, None, None

    # 检查解是否在有效范围内
    if 0 <= t <= 1 and 0 <= s <= 1:
        intersection_point = a1 + t * direction1
        return True, *intersection_point
    
    return False, None, None, None

######################################################\
    

##################################
#gpt
def has_mult_crossings2(crossings, proj):
    
    if not crossings:
        return None, None, None

    edge_to_idx = {}
    edges = []
    mult = {}
    proj_crossing = {}

    # 遍历crossings列表中的每个交叉点数据
    for crossing in crossings:
        edge1 = tuple(sorted([crossing[0], crossing[1]]))
        edge2 = tuple(sorted([crossing[2], crossing[3]]))
        
        if edge1 not in edge_to_idx:
            idx1 = len(edges)
            edge_to_idx[edge1] = idx1
            edges.append(edge1)
            mult[idx1] = []
            proj_crossing[idx1] = []
        else:
            idx1 = edge_to_idx[edge1]
        
        if edge2 not in edge_to_idx:
            idx2 = len(edges)
            edge_to_idx[edge2] = idx2
            edges.append(edge2)
            mult[idx2] = []
            proj_crossing[idx2] = []
        else:
            idx2 = edge_to_idx[edge2]

        if edge2 not in mult[idx1]:
            mult[idx1].append(edge2)
            TF, rx, ry, rz = intersect_3(proj[crossing[0]], proj[crossing[1]], proj[crossing[2]], proj[crossing[3]])
            proj_crossing[idx1].append([rx, ry, rz])
        
        if edge1 not in mult[idx2]:
            mult[idx2].append(edge1)
            TF, rx, ry, rz = intersect_3(proj[crossing[0]], proj[crossing[1]], proj[crossing[2]], proj[crossing[3]])
            proj_crossing[idx2].append([rx, ry, rz])

    return edges, mult, proj_crossing





#################################


def count_crossings_31(neigh_array, proj, is_closed):#3-维
    n = len(neigh_array)  
    res = []
    lasti = n
    order = {}

    for i in range(lasti):
        #print("i",i)
        if neigh_array[i][1] < 0:
            #print(neigh_array[i])
            continue
        lastj = n
        for j in range(i, lastj):
            #print('j', j)
            p2 = j % n
            #print("p2",p2)
            #if abs(i - p2) < 2:
                #continue
            if i == neigh_array[p2][1]:
                continue
            if neigh_array[i][1] == p2:
                continue
            if neigh_array[p2][1] < 0:
                continue
            intersect, rx, ry, rz = intersect_3(proj[i], proj[neigh_array[i][1]], proj[p2], proj[neigh_array[p2][1]])
            if intersect:
                if i not in order:
                    order[i] = [[i, neigh_array[i][1], p2, neigh_array[p2][1]]]
                else:
                    found = False
                    for z in range(len(order[i])):
                        if intersect_3(proj[order[i][z][0]], proj[order[i][z][1]], proj[order[i][z][2]], proj[order[i][z][3]]):
                            order[i].insert(z, [i, neigh_array[i][1], p2, neigh_array[p2][1]])
                            found = True
                            break
                    if not found:
                        order[i].append([i, neigh_array[i][1], p2, neigh_array[p2][1]])
                if i == neigh_array[p2][1]:
                    continue
                if neigh_array[i][1] == p2:
                    continue

    for i in order:
        for crossing in order[i]:
            res.append(crossing)

    #print(res)
    return res

#######################################


####################################

def process_coords(coords, proj, neigh_array, is_closed):
    #proj = project_vectors(coords)
    #print("投影后的点:\n", proj)
    n = len(neigh_array)
    before_cross = count_crossings_31(neigh_array, proj, is_closed)
    #print(before_cross)

    edge, mult, proj_crossing = has_mult_crossings2(before_cross, proj)
    #print(edge)
    if edge == None:
        return coords, proj, neigh_array
    
    for i in range(len(edge)):
        edge[i] = list(edge[i])
    #print(edge)

    #print(mult)
    #print(proj_crossing)

    new_proj = []
    new_coords = []
    new_neigh_array = []

    for k in range(n):
        new_proj.append(proj[k])
        new_coords.append(coords[k])
        new_neigh_array.append(neigh_array[k])

    se = {} 
    for i in range(len(edge)):
        se[i] = []
        if len(mult[i]) < 2:
            se[i].append(None)
        if len(mult[i]) > 1:
            for j in range(len(mult[i])):
                A1 = proj_crossing[i][j][0] - proj[edge[i][0]][0]
                A = proj[edge[i][1]][0] - proj[edge[i][0]][0]
                s = A1 / A
                se[i].append(s)
    #对se中的排序           
    for key in se:
    # 过滤掉None值，避免在排序时出错
        se[key] = sorted([x for x in se[key] if x is not None])
    #print('se\n', se) 

    coor = {}
    s_t_ = {}         
    for i in range(len(se)):
        coor[i] = []
        s_t_[i] = []
        if len(se[i]) > 1:
            for j in range(len(se[i]) - 1):
                s = (se[i][j] + se[i][j + 1]) * 0.5
                co = s * (np.array(coords[edge[i][1]]) - np.array(coords[edge[i][0]])) + np.array(coords[edge[i][0]])
                pr = s * (np.array(proj[edge[i][1]]) - np.array(proj[edge[i][0]])) + np.array(proj[edge[i][0]])
                s_t_[i].append(s)
                coor[i].append(co)
                new_coords.append(list(co))
                new_proj.append(pr)
    #print("coor\n",coor)           
    for i in range(len(coor)):
        if len(coor[i]) > 0:
        # 使用 np.array_equal 来比较数组
            coord_index = next((idx for idx, coord in enumerate(new_coords) if np.array_equal(coord, coor[i][0])), None)
            if coord_index is not None:
                new_neigh_array[edge[i][0]] = [new_neigh_array[edge[i][0]][0], coord_index]

            coord_index_last = next((idx for idx, coord in enumerate(new_coords) if np.array_equal(coord, coor[i][-1])), None)
            if coord_index_last is not None:
                new_neigh_array[edge[i][1]] = [coord_index_last, new_neigh_array[edge[i][1]][1]]
        
    # 剩下的处理方式保持不变
        if len(coor[i]) == 1:
            new_neigh_array.append([edge[i][0], edge[i][1]])
        if len(coor[i]) == 2: 
            coord_index = next((idx for idx, coord in enumerate(new_coords) if np.array_equal(coord, coor[i][1])), None)
            if coord_index is not None:
                new_neigh_array.append([edge[i][0], coord_index])

            coord_index_last = next((idx for idx, coord in enumerate(new_coords) if np.array_equal(coord, coor[i][0])), None)
            if coord_index_last is not None:
                new_neigh_array.append([coord_index_last, edge[i][1]])
                #print(coord_index_last)
        
        if len(coor[i]) > 2:
            coord_index_first = next((idx for idx, coord in enumerate(new_coords) if np.array_equal(coord, coor[i][1])), None)
            if coord_index_first is not None:
                new_neigh_array.append([edge[i][0], coord_index_first])
                #print(edge[i][0])

            for j in range(len(coor[i])-2):
                coord_index_start = next((idx for idx, coord in enumerate(new_coords) if np.array_equal(coord, coor[i][j])), None)
                coord_index_end = next((idx for idx, coord in enumerate(new_coords) if np.array_equal(coord, coor[i][j+2])), None)
                if coord_index_start is not None and coord_index_end is not None:
                    new_neigh_array.append([coord_index_start, coord_index_end])
                    #print(coord_index_start)
                    
                
            coord_index_last2 = next((idx for idx, coord in enumerate(new_coords) if np.array_equal(coord, coor[i][-2])), None)
            if coord_index_last2 is not None:
                #print(coord_index_last2)
                new_neigh_array.append([coord_index_last2, edge[i][1]])
                
        #print('nei',new_neigh_array)
            
              
        
    return new_coords, new_proj, new_neigh_array


def smoothing_0(neigh_array, coords, crossing):
    p1, p2, p3, p4 = crossing
    neigh_copy1 = [list(neigh) for neigh in neigh_array]
    #neigh_copy2 = [list(neigh) for neigh in neigh_array]
    if calculate_GL(coords[p1], coords[p2], coords[p3], coords[p4]) < 0:
        neigh_copy1[p1][1] = neigh_array[p3][1]
        neigh_copy1[p3][1] = neigh_array[p1][1]
        neigh_copy1[neigh_array[p1][1]][0] = p3
        neigh_copy1[neigh_array[p3][1]][0] = p1
    else:
        neigh_copy1[p1][1] = p3
        neigh_copy1[p3][1] = p1
        neigh_copy1[neigh_array[p1][1]][0] = neigh_array[p3][1]
        neigh_copy1[neigh_array[p3][1]][0] = neigh_array[p1][1]
    return neigh_copy1
        
        
def smoothing_1(neigh_array, coords, crossing):
    p1, p2, p3, p4 = crossing
    #neigh_copy1 = [list(neigh) for neigh in neigh_array]
    neigh_copy2 = [list(neigh) for neigh in neigh_array]
    if calculate_GL(coords[p1], coords[p2], coords[p3], coords[p4]) > 0:
        neigh_copy2[p1][1] = neigh_array[p3][1]
        neigh_copy2[p3][1] = neigh_array[p1][1]
        neigh_copy2[neigh_array[p1][1]][0] = p3
        neigh_copy2[neigh_array[p3][1]][0] = p1
    else:
        neigh_copy2[p1][1] = p3
        neigh_copy2[p3][1] = p1
        neigh_copy2[neigh_array[p1][1]][0] = neigh_array[p3][1]
        neigh_copy2[neigh_array[p3][1]][0] = neigh_array[p1][1]
    return neigh_copy2

def smooth(neigh_array, coords, crossings):
    n = len(crossings)
    combinations = list(itertools.product([-1, 1], repeat=n))
    state = []

    for j in range(len(combinations)):
        neigh_array1 = [row[:] for row in neigh_array]
        for i in range(n):
            #print(i)
            if combinations[j][i] == 1:
                neigh_array1 = smoothing_1(neigh_array1, coords, crossings[i])
            else:
                neigh_array1 = smoothing_0(neigh_array1, coords, crossings[i])
        state.append(neigh_array1)
    return state, combinations



########################################



####################################

def polylines(neigh_array):
    items_with_minus_one = [index for index, item in enumerate(neigh_array) if -1 in item]
    polylines = []
    
    for i in range(len(items_with_minus_one)):
        item = []
        a0 = -1
        a1 = items_with_minus_one[i]
        item.append(a1)
        
        a2 = neigh_array[a1].copy()  # 复制一份，避免直接修改原列表
        a2.remove(a0)
        a2 = a2[0]
        item.append(a2)
        
        while a2 not in items_with_minus_one:
            a0 = a1
            a1 = a2
            a2 = neigh_array[a1].copy()
            #print(a2,a0)
            a2.remove(a0)
            
            a2 = a2[0]
            item.append(a2)
        
        # 添加起点和终点
        polylines.append([item[0], item[-1]])
    
    return polylines


def pair_(a, polylines):
    for item in polylines:
        if item[0] == a:
            a1 = item[1]
            break
    return a1

def segment_cycles(polylines0, polylines1, neigh_array):
    items_with_minus_one = [index for index, item in enumerate(neigh_array) if -1 in item]

    state_cycles = []
    cycle = []
    for item in items_with_minus_one:
        
        if item not in cycle:
            cycle = []
            a0 = item
            cycle.append(a0)

            a1 = pair_(a0, polylines1)
            cycle.append(a1)

            a2 = pair_(a1, polylines0)
            cycle.append(a2)

            while a2 != a0:
                a1 = pair_(a2, polylines1)
                cycle.append(a1)

                a2 = pair_(a1, polylines0)
                cycle.append(a2)
            state_cycles.append(cycle)
    return state_cycles


#################################

###################################



def mult_poly_d(power):
    """
    Multiply two polynomials, used in calculating the Jones polynomial

    :param power: The number of separate loops in the system being measured
    :return: The polynomial of a single combination of crossing annealments
    """
    base = {2: -1, -2: -1}
    if power == 0:
        return {0: 1}
    if power == 1:
        return base

    result = {}
    temp = {2: -1, -2: -1}
    for k in range(2, power+1):
        for x_key, x_val in temp.items():
            for y_key, y_val in base.items():
                if x_key + y_key in result:
                    result[x_key + y_key] += x_val * y_val
                else:
                    result[x_key + y_key] = x_val * y_val
        temp = result
        result = {}
#注意power=1的时候需要处理
    return temp

def simple_add(a, b):
    """
    Add two polynomials where both are passed as parameters.
    
    :param a: The first polynomial represented as a dictionary {power: coefficient}.
    :param b: The second polynomial represented as a dictionary {power: coefficient}.
    :return: The sum of a and b represented as a dictionary {power: coefficient}.
    """
    result = {}

    # Add terms from the first polynomial
    for power, coeff in a.items():
        result[power] = coeff

    # Add terms from the second polynomial
    for power, coeff in b.items():
        if power in result:
            result[power] += coeff
        else:
            result[power] = coeff

    return result

def calculate_Wr(crossings, coords):
    wr = 0
    for crossing in crossings:
        a1 = coords[crossing[0]]
        a2 = coords[crossing[1]]
        b1 = coords[crossing[2]]
        b2 = coords[crossing[3]]
        if calculate_GL(a1, a2, b1, b2) > 0:
            wr += 1
            #print('+')
        else :
            wr -= 1
            #print('-')
    return wr
            

#############################

def Jones(coords, neigh_array, crossings):
    # 计算 wr 只需要一次
    wr = calculate_Wr(crossings, coords)

    # 获取所有平滑后的状态及其组合
    state, combinations = smooth(neigh_array, coords, crossings)

    # 预计算多项式A和初始多段线polylines0
    monial_A = []
    #print(neigh_array)
    polylines0 = polylines(neigh_array)
    
    for combination in combinations:
        monial1 = {0: 1}    
        for i in combination:
            monial1 = simple_mult(monial1, {i: 1})
        monial_A.append(monial1)

    loop_numbers = []
    segments = []

    # 计算每个状态的 loop number 和 segments
    for state0 in state:
        polylines1 = polylines(state0)
        segment1 = segment_cycles(polylines0, polylines1, neigh_array)
        segments.append(segment1)
        loop_numbers.append(count_loops(state0))
    
    # 计算 monial_S_cyc_circ_1
    monial_S_cyc_circ_1 = [
        mult_poly_d(loop_numbers[i] - 1 + len(segments[i]))
        for i in range(len(state))
    ]

    # 计算所有的多项式 monial_all
    monial_all = [
        simple_mult(monial_A[i], monial_S_cyc_circ_1[i])
        for i in range(len(state))
    ]

    # 将所有的多项式相加得到 monial_bracket
    monial_bracket = {}
    for monial in monial_all:
        monial_bracket = simple_add(monial_bracket, monial)

    # 计算最终的 Jones 多项式
    monial_Jones = simple_mult({(-wr)*3: (-1)**(-wr)}, monial_bracket)
    #print(monial_Jones)
    return monial_Jones


##########################



def Jones_3_dim (coords, neigh_array, num_proj):
    avejones = defaultdict(float)
    num_fails = 0
    #init_n = n
    tot = 0
    
    for z in range(num_proj):
        proj = project_vectors(coords)
        new_coords, new_proj, new_neigh_array = process_coords(coords, proj, neigh_array, False)
        crossings = count_crossings_31(new_neigh_array, new_proj, False)
        if len(crossings) == 0:
            continue
        if len(crossings) > 15:
            num_fails += 1
            if num_fails >= 25:
                return {0: 0}
            continue
        #调试
        #print("proj",proj)
        #print("nei",new_neigh_array)
        #print("newproj",new_proj)
        jones = Jones(new_coords, new_neigh_array, crossings)
        tot += 1
        
        for k, v in jones.items():
            if v != 0:
                avejones[k] += v
        #print(avejones)
                
    final = {}
    for k, v in avejones.items():
        final[k] = v / tot
    #print(final)
    
    final_jones = {}
    for k, v in final.items():
        final_jones[k*(-0.25)] = v
    #print(final_jones)
    return final_jones



