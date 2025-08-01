# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:41:10 2024

@author: Lenovo
"""
import gudhi
import Jones_3_dim as J
import numpy as np
from scipy.spatial import distance_matrix
from Bio import PDB
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import math
import os
import timeit

def get_protein_ca_atom_coordinate(pdbid, filepath):
    parser = PDB.PDBParser()
    struct = parser.get_structure(pdbid, filepath)

    CA_coordinates = np.array([])

    for model in struct:
        coor = []
        labels = np.array([])
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.id == "CA":
                        XYZ = atom.get_coord()
                        bfactor = atom.bfactor
                        labels = np.append(labels, bfactor)
                        CA_coordinates = np.hstack([CA_coordinates, XYZ])

        break

    return CA_coordinates.reshape(-1, 3), labels


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
            a2.remove(a0)
            a2 = a2[0]
            item.append(a2)
        
        # 添加起点和终点
        polylines.append([item[0], item[-1]])
    
    return polylines


def filter_points_with_indices(points, reference_point, max_distance):
    """
    从点云数据中筛选出距离参考点小于给定距离的所有点，以及这些点在原始数据中的索引。

    参数:
    - points: 一个Numpy数组，形状为 (n, 3)，表示 n 个三维点的坐标。
    - reference_point: 一个列表或元组，表示参考点的 (x, y, z) 坐标。
    - max_distance: 一个浮点数，表示筛选的最大距离阈值。

    返回:
    - 筛选出的点的Numpy数组，和对应的原始数据中的索引列表。
    """
    # 计算每个点到参考点的欧几里得距离
    distances = np.linalg.norm(points - reference_point, axis=1)
    
    # 找出距离小于 max_distance 的点的索引
    indices = np.where(distances < max_distance)[0]
    
    # 筛选出这些点
    filtered_points = points[indices]
    
    return filtered_points, indices

def filter_points_with_indices_int_(points, reference_point, min_distance, max_distance):
    """
    筛选出到参考点距离大于min_distance，小于max_distance 的点及其索引，
    同时将参考点及其索引添加到集合中。

    参数:
        points (np.ndarray): 点云数据，形状为 (N, 3)，每一行表示一个点的 (x, y, z) 坐标。
        reference_point (np.ndarray): 参考点，形状为 (3, )，表示参考点的 (x, y, z) 坐标。
        min_distance (float): 最小距离。
        max_distance (float): 最大距离。

    返回:
        tuple: 两个列表，第一个列表包含筛选出来的点，第二个列表包含对应的索引。
    """
    filtered_points = []
    indices = []
    
    for i, point in enumerate(points):
        distance = np.linalg.norm(point - reference_point)
        
        if min_distance < distance < max_distance:
            filtered_points.append(point)
            indices.append(i)
    
    # 添加参考点及其索引
    filtered_points.append(reference_point)
    indices.append(len(points))
    
    return filtered_points, indices

def filter_points_int(points, reference_point, min_distance, max_distance):
    """
    筛选出到参考点距离大于min_distance且小于max_distance的点及其指标，
    并将参考点及其在点云数据中的指标添加到筛选出来的集合中。
    
    参数:
    points (np.ndarray): 点云数据，形状为(N, 3)。
    reference_point (np.ndarray): 参考点，形状为(3,)。
    min_distance (float): 最小距离。
    max_distance (float): 最大距离。
    
    返回:
    filtered_points (np.ndarray): 筛选出来的点，形状为(M, 3)。
    filtered_indices (np.ndarray): 筛选出来的点的指标，形状为(M,)。
    """
    # 计算所有点到参考点的距离
    distances = np.linalg.norm(points - reference_point, axis=1)
    
    # 筛选出符合条件的点的索引
    mask = (distances > min_distance) & (distances < max_distance)
    filtered_points = points[mask]
    filtered_indices = np.where(mask)[0]
    
    # 将参考点及其在点云数据中的指标添加到筛选出来的集合中
    filtered_points = np.vstack([filtered_points, reference_point])
    reference_index = np.where((points == reference_point).all(axis=1))[0][0]
    filtered_indices = np.append(filtered_indices, reference_index)
    
    sorted_indices = np.argsort(filtered_indices)
    filtered_indices = filtered_indices[sorted_indices]
    filtered_points = filtered_points[sorted_indices]
    
    return filtered_points, filtered_indices

def filter_points_with_indices_int(points, reference_point, max_distance, min_distance):
    """
    从点云数据中筛选出距离参考点小于给定距离的所有点，以及这些点在原始数据中的索引。

    参数:
    - points: 一个Numpy数组，形状为 (n, 3)，表示 n 个三维点的坐标。
    - reference_point: 一个列表或元组，表示参考点的 (x, y, z) 坐标。
    - max_distance: 一个浮点数，表示筛选的最大距离阈值。

    返回:
    - 筛选出的点的Numpy数组，和对应的原始数据中的索引列表。
    """
    # 计算每个点到参考点的欧几里得距离
    distances = np.linalg.norm(points - reference_point, axis=1)
    
    #reference_index = np.where((points == reference_point).all(axis=1))[0][0]
    # 找出距离小于 max_distance 的点的索引
    
    indices = np.where((distances >= min_distance) & (distances < max_distance))[0]
    
    # 筛选出这些点
    filtered_points = points[indices]
    
    return filtered_points, indices

def find_consecutive_indices(indices):
    """
    找出索引列表中连续的数字序列。

    参数:
    - indices: 一个包含整数的列表或数组。

    返回:
    - 一个列表，其中每个元素是一个包含连续数字的子列表。
    """
    # 确保输入是排序好的
    if len(indices) == 0:
        return []
    
    indices = sorted(indices)
    
    # 用于存储结果的列表
    result = []
    current_group = [indices[0]]

    # 遍历索引列表，寻找连续的序列
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_group.append(indices[i])
        else:
            result.append(current_group)
            current_group = [indices[i]]
    
    # 添加最后一个分组
    result.append(current_group)
    
    return result

#coords, label = get_protein_ca_atom_coordinate('pdbid', "1V70.pdb")
#filtered_points, filtered_indices = filter_points_int(coords, coords[88], 4,6)
#lines = find_consecutive_indices(filtered_indices)

def generate_neigh_array(n, closed):
    """
    创建一个包含每个原子邻居索引的列表的列表。
    例如：neigh_array[1] 包含原子1的邻居，它们是0和2。

    参数:
    - n: 系统中原子的数量
    - closed: 如果系统中的最后一个原子与第一个原子相连，则为True，否则为False

    返回:
    - 一个包含每个原子邻居的列表的列表
    """
    neigh_array = []
    
    # 中间原子的邻居关系
    for i in range(1, n - 1):
        neigh_array.append([i - 1, i + 1])
    
    # 特殊处理第一个和最后一个原子的邻居关系
    if closed:
        neigh_array.append([n - 2, 0])
        neigh_array.insert(0, [n - 1, 1])
    else:
        neigh_array.append([n - 2, -1])
        neigh_array.insert(0, [-1, 1])
    
    return neigh_array

# 示例用法

def merge_and_shift_neigh_arrays(array1, array2):
    """
    合并两个邻居数组，并使得第二个数组中的大于等于0的数字都加上 len(array1)。
    
    参数:
    - array1: 第一个邻居数组
    - array2: 第二个邻居数组
    
    返回:
    - 合并后的邻居数组
    """
    len_array1 = len(array1)
    
    # 调整array2中大于等于0的索引
    shifted_array2 = []
    for neighbors in array2:
        shifted_neighbors = [(x + len_array1) if x >= 0 else x for x in neighbors]
        shifted_array2.append(shifted_neighbors)
    
    # 合并两个数组
    merged_array = array1 + shifted_array2
    
    return merged_array



def far_phe(pdbid, filepath, min_distance, max_distance, times):
    coords, label = get_protein_ca_atom_coordinate(pdbid, filepath)
    #print(len(coords))

    jones_ca = []
    lines_ca = []
    feature_ca = []
    phe_ca = []
    h = 0
    for o in range(len(coords)):
        filter_points , indices = filter_points_int(coords, coords[o], min_distance, max_distance)
        #print(indices)
        indices = indices[indices != o]
        #print(indices)
        #print(indices)
        phe__ = 0
        for indice in indices:
            distances = np.linalg.norm(coords[indice] - coords[o], axis=0)
            phe__ += 1/(1+(distances/8))
        #print(phe__)
        phe_ca.append(phe__)
    #np.save('far_phe_'+ filepath +str(min_distance)+'_'+ str(max_distance) +'.npy', phe_ca)
    return phe_ca
''' 
for r in range(5,28):
    print("r:",r)
    phe_ca = far_phe('pdbid', "1V70.pdb", r-1, r, 50)      
'''
#features_46 = [np.load('far_phe4_6.npy')]

def jones_ca_int(pdbid, filepath, min_distance, max_distance, times):
    coords, label = get_protein_ca_atom_coordinate(pdbid, filepath)
    #print(len(coords))

    jones_ca = []
    lines_ca = []
    feature_ca = []
    h = 0
    for o in range(len(coords)):
        filter_points , indices = filter_points_int(coords, coords[o], min_distance, max_distance)
       
        #print(indices)
       
        
        lines = find_consecutive_indices(indices)
        #print(lines)
        lines_ca.append(lines)
        neigh_array = []
        for line in lines:
            
            neigh_array1 = generate_neigh_array(len(line)+2, False)
            
            neigh_array = merge_and_shift_neigh_arrays(neigh_array, neigh_array1)
            
        filter_points_lines = filter_points
        i = 0
        #print(filter_points_lines)
        for line in lines:
            #print(line)
            if line[0] == 0:
                start_point = (coords[line[0]] + coords[line[0]])*0.5
                filter_points_lines = np.insert(filter_points_lines, i, start_point, axis=0)
                i = i + len(line) +1
                #print("line", i)
                end_point = (coords[line[-1]] + coords[line[-1]+1])*0.5
                filter_points_lines = np.insert(filter_points_lines, i, end_point, axis=0)
                i += 1
                #print(filter_points_lines)
            elif line[-1] ==len(coords)-1:
                start_point = (coords[line[0]-1] + coords[line[0]])*0.5
                filter_points_lines = np.insert(filter_points_lines, i, start_point, axis=0)
                i = i + len(line) +1
                end_point = (coords[line[-1]] + coords[line[-1]])*0.5
                filter_points_lines = np.insert(filter_points_lines, i, end_point, axis=0)
                i += 1
            else:
                #print(i)
                start_point = (coords[line[0]-1] + coords[line[0]])*0.5
                filter_points_lines = np.insert(filter_points_lines, i, start_point, axis=0)
                i = i + len(line) +1
                #print(filter_points_lines)
                end_point = (coords[line[-1]] + coords[line[-1]+1])*0.5
                filter_points_lines = np.insert(filter_points_lines, i, end_point, axis=0)
                #print(filter_points_lines)
                i += 1
        
        jones = J.Jones_3_dim(filter_points_lines, neigh_array, times)
        feature = 0
        t = 10
        for k,v in jones.items():
            feature1 = (t**k)*v
            feature = feature + feature1
        feature_ca.append(feature)
        jones_ca.append(jones)
        h += 1
        #print(h)
    #np.save('feature_ca_int_'+ filepath +str(min_distance)+'_'+ str(max_distance) +'.npy', feature_ca)
    #np.save('jones_ca_int_'+ filepath +str(min_distance)+'_'+ str(max_distance) +'.npy', jones_ca)
    return feature_ca, jones_ca
'''
for r in range(6,7):
    print("r:",r)
    feature_ca, jones_ca = jones_ca_int('pdbid', "1V70.pdb", r-1, r, 50)
'''
def Gauss_linking_integral(a1,a2, b1,b2):
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

#gauss1 = Gauss_linking_integral([0,0,0],[1,1,1], [1,0,1], [0,1,1])
#gauss2 = J.calculate_GL_([0,0,0],[1,1,1], [1,0,1], [0,1,1])

def gauss_link(pdbid, filepath, min_distance, max_distance, times):
    coords, label = get_protein_ca_atom_coordinate(pdbid, filepath)
    #print('coords',len(coords))

    jones_ca = []
    lines_ca = []
    feature_ca = []
    phe_ca = []
    gauss_ca = []
    h = 0
    for o in range(len(coords)):
        #print(o)
        filter_points , indices = filter_points_int(coords, coords[o], min_distance, max_distance)
       
        #print(indices)
        indices = indices[indices != o]
        #print(indices)
        gauss_ = 0
        
        if o == 0:
            a1 = coords[o]
            a2 = (coords[o]+coords[o])*0.5
            a3 = coords[o]
            a4 = (coords[o+1]+coords[o])*0.5 
        elif o == len(coords)-1:
            a1 = coords[o]
            a2 = (coords[o-1]+coords[o])*0.5
            a3 = coords[o]
            a4 = (coords[o]+coords[o])*0.5 
        else:
            a1 = coords[o]
            a2 = (coords[o-1]+coords[o])*0.5
            a3 = coords[o]
            a4 = (coords[o+1]+coords[o])*0.5 
            

        for indice in indices:
            #print(indice)
            
            if indice == 0 :
                b1 = coords[indice]
                b2 = (coords[indice]+ coords[indice])*0.5
                c1 = coords[indice]
                c2 = (coords[indice+1]+ coords[indice])*0.5
            elif indice ==len(coords)-1:
                b1 = coords[indice]
                b2 = (coords[indice-1]+ coords[indice])*0.5
                c1 = coords[indice]
                c2 = (coords[indice]+ coords[indice])*0.5
            else:
                b1 = coords[indice]
                b2 = (coords[indice-1]+ coords[indice])*0.5
                c1 = coords[indice]
                c2 = (coords[indice+1]+ coords[indice])*0.5
            
            gauss1 = J.calculate_GL(a1, a2, b1, b2)
            gauss2 = J.calculate_GL(a1, a2, c1, c2)
            gauss3 = J.calculate_GL(a3, a4, c1, c2)
            gauss4 = J.calculate_GL(a3, a4, b1, b2)
            gauss_ += gauss1
            gauss_ += gauss2
            gauss_ += gauss3
            gauss_ += gauss4
        gauss_ca.append(gauss_)
        #print(gauss_)
                   
    #np.save('gauss_link_'+ filepath +str(min_distance)+'_'+ str(max_distance) +'.npy', gauss_ca)
    #np.save('gauss_link_' + filepath + str(min_distance) + '_' + str(max_distance) + '.npy', gauss_ca)
    return gauss_ca
'''
for r in range(5,6):
    print("rrrrrrrrrrrrrrrrrrr:",r)
    gauss_link_ca_ = gauss_link('pdbid', "1V70.pdb", r-1, r, 50)
'''
def normalize_feature(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)

def fitting(r, X_norm, y, model):
    regressor = model.fit(X_norm, y)
    y_pred = regressor.predict(X_norm)
    return pearsonr(y_pred, y)[0]

def generate_feature(filepath):
    
    folder_name = os.path.splitext(filepath)[0]  # 使用文件名（不含扩展名）作为文件夹名
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    for r in range(6,10):
        r_ = r/1
        print("rrrrrrrrrrrrrrrrrrr:", r_)
        
        gauss_link_ca_ = gauss_link('pdbid', filepath, r_ - 1, r_, 50)
        feature_ca, jones_ca = jones_ca_int('pdbid', filepath, r_ - 1, r_ , 50)
        phe_ca = far_phe('pdbid', filepath, r_ -1, r_ , 50)  
        
        np.save(os.path.join(folder_name, 'gauss_link_' + str(r_ - 1) + '_' + str(r_) + '.npy'), gauss_link_ca_)
        np.save(os.path.join(folder_name, 'jones_feature_' + str(r_ - 1) + '_' + str(r_) + '.npy'), feature_ca)
        np.save(os.path.join(folder_name, 'jones_polynomial_' + str(r_ - 1) + '_' + str(r_) + '.npy'), jones_ca)
        np.save(os.path.join(folder_name, 'fri_' + str(r_ - 1) + '_' + str(r_) + '.npy'), phe_ca)


#generate_feature('b_factor_data_small/2olx.pdb')
def _main_ ():
    ppbd_text = open('medium_ID.txt')
    pro_pbd_id = ppbd_text.readlines()
    pbd_id = eval(pro_pbd_id[0])
    ppbd_text.close()

    for pbd in range(0,1):
        #index = pdb_list.index('1p9i')
        print(pbd)
        print(pbd_id[pbd])
        generate_feature('b_factor_data_medium/'+ pbd_id[pbd] +'.pdb')

execution_time = timeit.timeit(_main_, number=1)
print(f"运行时间: {execution_time} 秒")



pdb_text = open('medium_ID.txt')
pro_pdb_id = pdb_text.readlines()
pdb_id = eval(pro_pdb_id[0])
pdb_text.close()
b_factor_pdb_pvalue = []

for pdb in range(0,1):
    folder_name = 'b_factor_data_medium/'+ pdb_id[pdb]
    coords, label = get_protein_ca_atom_coordinate('pdbid', folder_name + '.pdb')
    pro_jones_feature = []
    pro_fri_feature = []
    pro_gauss_feature = []
    for r in range(6, 10):
        r_ = r/1
        print("rrrrrrrrrrrrrrrrrrr:", r_)  
         
        jones = np.load(os.path.join(folder_name, 'jones_feature_' + str(r_ - 1) + '_' + str(r_) + '.npy'))
        pro_jones_feature.append(jones)
        fri = np.load(os.path.join(folder_name, 'fri_' + str(r_ - 1) + '_' + str(r_) + '.npy'))
        pro_fri_feature.append(fri)
        gauss = np.load(os.path.join(folder_name, 'gauss_link_' + str(r_ - 1) + '_' + str(r_) + '.npy'))
        pro_gauss_feature.append(gauss)
        
    jones_feature = np.zeros((len(coords),len(pro_jones_feature)))
    fri_feature = np.zeros((len(coords),len(pro_jones_feature)))
    gauss_feature = np.zeros((len(coords),len(pro_jones_feature)))
    for i in range(len(jones_feature)):
        for j in range(len(pro_jones_feature)):
            jones_feature[i][j] =  1 / (abs(pro_jones_feature[j][i]) + 1e-4)#math.e **-( abs(pro_jones_feature[j][i] ) ) 
            fri_feature[i][j] =  abs(pro_fri_feature[j][i]) 
            gauss_feature[i][j] =  1 / (abs(pro_gauss_feature[j][i]) + 1e-4)
            
    x_norm_1 = normalize_feature(jones_feature)
    x_norm_2 = normalize_feature(fri_feature)
    x_norm_3 = normalize_feature(gauss_feature)
    model = LinearRegression()
    r = 27
    pvalue = fitting(r, x_norm_1, label, model)

    b_factor_pdb_pvalue.append(pvalue)
    
mean_value = np.nanmean(b_factor_pdb_pvalue)

average = sum(b_factor_pdb_pvalue) / len(b_factor_pdb_pvalue)




