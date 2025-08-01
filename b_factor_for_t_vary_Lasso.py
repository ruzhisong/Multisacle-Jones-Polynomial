# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:30:44 2024

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
from sklearn.linear_model import Lasso
from scipy.stats import pearsonr
import math
import os
import timeit
from sklearn import tree
from sklearn.datasets import load_iris

def get_protein_ca_atom_coordinate(pdbid, filepath):
    parser = PDB.PDBParser()
    struct = parser.get_structure('protein', filepath)

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

def normalize_feature(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)

def fitting(X_norm, y, model):
    regressor = model.fit(X_norm, y)
    y_pred = regressor.predict(X_norm)
    return pearsonr(y_pred, y)[0]

def average_pvalue_of_data_set(data_set):
    with open(data_set + '_ID.txt', 'r') as pdb_text:
        pro_pdb_id = pdb_text.readlines()
    pdb_id = eval(pro_pdb_id[0])

    b_factor_pdb_pvalue = []

    for pdb in range(len(pdb_id)):
        folder_name = 'b_factor_data_' + data_set + '/' + pdb_id[pdb]
        print(folder_name)
        coords, label = get_protein_ca_atom_coordinate('pdbid', folder_name + '.pdb')
        pro_jones_feature = []
        pro_fri_feature = []
        pro_gauss_feature = []
        pro_jones_polynomial = []

        for r in range(20, 65):
            r_ = r / 4
            print("rrrrrrrrrrrrrrrrrrr:", r_)

            jones = np.load(os.path.join(folder_name, 'jones_feature_' + str(r_ - 1) + '_' + str(r_) + '.npy'),
                            allow_pickle=True)
            pro_jones_feature.append(jones)
            polynomial = np.load(os.path.join(folder_name, 'jones_polynomial_' + str(r_ - 1) + '_' + str(r_) + '.npy'),
                                 allow_pickle=True)
            pro_jones_polynomial.append(polynomial)
            # fri = np.load(os.path.join(folder_name, 'fri_' + str(r_ - 1) + '_' + str(r_) + '.npy'), allow_pickle=True)
            # pro_fri_feature.append(fri)
            # gauss = np.load(os.path.join(folder_name, 'gauss_link_' + str(r_ - 1) + '_' + str(r_) + '.npy'), allow_pickle=True)
            # pro_gauss_feature.append(gauss)

        jones_feature = np.zeros((len(coords), len(pro_jones_feature)))
        jones_polynomial_5 = np.zeros((len(coords), len(pro_jones_polynomial)))
        jones_polynomial_10 = np.zeros((len(coords), len(pro_jones_polynomial)))
        jones_polynomial_15 = np.zeros((len(coords), len(pro_jones_polynomial)))
        # fri_feature = np.zeros((len(coords), len(pro_jones_feature)))
        # gauss_feature = np.zeros((len(coords), len(pro_jones_feature)))

        for i in range(len(jones_feature)):
            for j in range(len(pro_jones_feature)):
                jones_feature[i][j] = 1 / (abs(pro_jones_feature[j][i]) + 1e-4)
                # print(len(pro_jones_polynomial[j][i]))
                # jones_polynomial[i][j] = pro_jones_polynomial[j][i]
                # jones_ca = []
                # lines_ca = []
                feature_ca = []
                # h = 0
                feature_5 = 0
                feature_10 = 0
                feature_15 = 0
                t1 = 5
                t2 = 10
                t3 = 15
                for k, v in pro_jones_polynomial[j][i].items():
                    # print(k,v)
                    feature_51 = (t1 ** k) * v
                    feature_5 = feature_5 + feature_51
                    feature_101 = (t2 ** k) * v
                    feature_10 = feature_10 + feature_101
                    feature_151 = (t3 ** k) * v
                    feature_15 = feature_15 + feature_151
                jones_polynomial_5[i][j] = 1 / (abs(feature_5) + 1e-4)
                jones_polynomial_10[i][j] = 1 / (abs(feature_10) + 1e-4)
                jones_polynomial_15[i][j] = 1 / (abs(feature_15) + 1e-4)

                # fri_feature[i][j] = 1 / (abs(pro_fri_feature[j][i]) + 1e-4)
                # gauss_feature[i][j] = 1 / (abs(pro_gauss_feature[j][i]) + 1e-4)

        x_norm_1 = normalize_feature(jones_polynomial_5)
        x_norm_2 = normalize_feature(jones_polynomial_10)
        x_norm_3 = normalize_feature(jones_polynomial_15)
        x_norm = np.hstack(( x_norm_1, x_norm_2))

        # 使用 Lasso 回归
        model = Lasso(alpha=0.16)
        pvalue = fitting(x_norm_2, label, model)
        b_factor_pdb_pvalue.append(pvalue)

    mean_value = np.nanmean(b_factor_pdb_pvalue)
    average = sum(b_factor_pdb_pvalue) / len(b_factor_pdb_pvalue)

    return mean_value, average, x_norm_2, b_factor_pdb_pvalue

s_average = average_pvalue_of_data_set('small')
m_average = average_pvalue_of_data_set('medium')
l_average = average_pvalue_of_data_set('large')
