#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Author: JeffreyMa
# -----
# Copyright (c) 2023 iFLYTEK & USTC
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###

import os
from typing import List, Dict
from apted import APTED, Config

__all__ = ['generate_doc_tree_from_log_line_level', 'tree_edit_distance', 'vis_digraph_py', 'complete_json']

Relation_Color = {"contain": "red", "equality": "blue", "connect": "purple"}
Page_Color = ["antiquewhite", "cadetblue1", "brown1", "azure2", "chartreuse", "chocolate1"]

def tree_edit_distance(pred_tree, true_tree):
    distance = APTED(pred_tree, true_tree, Config()).compute_edit_distance()
    teds = 1.0 - (float(distance) / max([len(pred_tree), len(true_tree)]))
    return distance, teds

class Node():
    def __init__(self, name, info=None):
        """
        name: unique id for this node
        children, parent: real in doc_tree
        ref_children, ref_parent: model output for each node
        """
        self.name = name
        self.info = info
        self.children = list()
        self.parent = None
        self.ref_children = list()
        self.ref_parent = None
        self.ref_parent_relation = None
        self.depth = None

    def _set_parent(self, node):
        self.parent = node

    def _set_ref_parent(self, node, relation):
        self.ref_parent = node
        self.ref_parent_relation = relation

    def add_child(self, node):
        self.children.append(node)
        node._set_parent(self)

    def add_ref_child(self, node, relation):
        self.ref_children.append(node)
        node._set_ref_parent(self, relation)

    def set_depth(self, cur_depth):
        self.depth = cur_depth
        for child in self.children:
            child.set_depth(cur_depth + 1)

    def generate_repr(self, output_lst):
        cur_repr = '\t' * self.depth + self.name
        output_lst.append(cur_repr)
        for child in self.children:
            child.generate_repr(output_lst)

    def __repr__(self):
        if self.name == 'ROOT':
            output_str_lst = []
            self.generate_repr(output_str_lst)
            return '\n'.join(output_str_lst)
        else:
            return self.name

    def __len__(self):
        length = 1
        for child in self.children:
            length += len(child)
        return length

def generate_doc_tree_from_log_line_level(boxes_texts, parent_idxs, relations):
    """
    1.one `Node` for each text box, add ROOT node
    2.set ref_parent and ref_children for each text box given model output and transform ref_child to child at once,
      meta nodes point to root directly.
    3.denote this tree with its root node
    Note:
    1.`relative_parent_idx` is different from `generate_doc_tree_from_gt` in step 2
    2.some rules are added in step 2 when transform ref_child to child
    """
    assert len(boxes_texts) == len(parent_idxs) == len(relations)
    # step 1
    node_lst =[Node(name='ROOT')]
    for text in boxes_texts:
        node = Node(name=text)
        node_lst.append(node)
    # step 2
    for b_i, box in enumerate(node_lst):
        if b_i > 0:
            no_root_idx = b_i - 1 # remove root
            if parent_idxs[no_root_idx] == 0:
                relative_parent_idx = parent_idxs[no_root_idx] + 1
            else:
                relative_parent_idx = parent_idxs[no_root_idx] + 1
            relation = relations[no_root_idx]
            parent = node_lst[relative_parent_idx]
            parent.add_ref_child(box, relation)
            if relation in ['contain', 'connect']:
                box.ref_parent.add_child(box)
            if relation == 'equality':
                oldest_bro = box.ref_parent
                while oldest_bro.ref_parent_relation == 'equality':
                    oldest_bro = oldest_bro.ref_parent
                if oldest_bro.parent:
                    oldest_bro.parent.add_child(box)

    root = node_lst[0]
    root.set_depth(cur_depth=0) # set depth for better visualize
    return root

def vis_digraph_py(json_data: List[Dict], out_folder: str, dig_name: str="doc_vis", save_image: bool=True):
    """The python version of vis_digraph, recommanded.

    Args:
        json_data ([dict]):  [the json data of document elements]
        out_folder ([str]):  [the path to store the .dot file and .pdf file]
        dig_name   ([str]):  [the name of vis file]
    """

    """Parameter Validation"""
    cl_ids = [x["line_id"] for x in json_data]
    if len(cl_ids) != len(set(cl_ids)):
        from collections import Counter
        x = dict(Counter(cl_ids))
        dup = {key:value for key,value in x.items()if value > 1}
        print("json_data invalid! line ids are duplicated : {}".format(repr(dup)))
        raise ValueError
    if not os.path.exists(out_folder):
        print("out_folder not exists! trying to make one.")
        os.makedirs(out_folder)
    if not save_image:
        print("Assign save_image=False will incur not saving any doc tree!")

    import graphviz
    dot = graphviz.Digraph(dig_name, comment=dig_name+" dot file")
    """Process Json File, Generate Doc Tree"""
    for cl in json_data:
        content  = str(cl["line_id"])
        dot.node(str(cl["line_id"]), content, \
            color=Page_Color[cl["page"]%len(Page_Color)], style='filled')
        dot.edge(str(cl["parent_id"]), str(cl["line_id"]), \
            color=Relation_Color[cl["relation"]])
    if save_image:
        dot.render(format='pdf', directory=out_folder)

def complete_json(pred_info, gt_info):
    generated_json = []
    for idx, item in enumerate(pred_info):
        item["line_id"] = idx
        item["page"] = gt_info[idx]["page"]
        generated_json.append(item)
    return generated_json

def transfer_tree_to_chain(tree):
    chain = []
    # DFS to traverse the tree
    def dfs(node):
        chain.append(node.name)
        if node.children.__len__() == 0 and 'section' not in node.name:
            chain.append('<p>')
            return
        elif node.children.__len__() == 0 and 'section' in node.name:
            return
        for child in node.children:
            if child.name[:6] == 'figure' or child.name[:7] == 'caption' or child.name[:5] == 'table':
                continue
            dfs(child)
    
    dfs(tree)
    
    chain_floating = []
    # DFS to traverse the tree
    def dfs_floating(node, level=1):
        chain_floating.append(node.name)
        for child in node.children:
            if child.name[:6] == 'figure' or child.name[:7] == 'caption' or child.name[:5] == 'table':
                dfs_floating(child, level+1)
            else:
                continue
        if level == 2:
            chain_floating.append('<p>')
            return
    
    dfs_floating(tree)
    return chain, chain_floating

def edit_distance(chain1, chain2):  
    m, n = len(chain1), len(chain2)  
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]  
  
    # 初始化边界值  
    for i in range(m + 1):  
        dp[i][0] = i  
    for j in range(n + 1):  
        dp[0][j] = j  
  
    # 动态规划计算编辑距离  
    for i in range(1, m + 1):  
        for j in range(1, n + 1):  
            if chain1[i - 1] == chain2[j - 1]:  
                substitution_cost = 0  
            else:  
                substitution_cost = 1  
  
            dp[i][j] = min(dp[i - 1][j] + 1,  # 删除操作  
                           dp[i][j - 1] + 1,  # 插入操作  
                           dp[i - 1][j - 1] + substitution_cost)  # 替换操作  
  
    edit_distance = dp[m][n]  
    max_possible_distance = max(m, n)  
    edit_score = 1 - edit_distance / max_possible_distance  
  
    return edit_distance, edit_score

import numpy as np
from scipy.optimize import linear_sum_assignment

def min_edit_distance_between_lists(list1, list2):  
    max_distance = max(len(sum(list1, [])), len(sum(list2, [])))
    if max_distance == 0:
        return 0.0, 1.0
    matrix = np.zeros((len(list1), len(list2)))
    matrix_distance = matrix.copy()
      
    for i, chain1 in enumerate(list1):  
        for j, chain2 in enumerate(list2):  
            matrix_distance[i][j], matrix[i][j] = edit_distance(chain1, chain2)
  
    # row_indices, col_indices = linear_sum_assignment(matrix, maximize=True)
    # total_scores = matrix[row_indices, col_indices].sum()
    # total_distances = matrix_distance[row_indices, col_indices].sum()
    # return total_distances, total_scores / max(len(list1), len(list2))
    # if len(list1) > len(list2):
    #     row_indices, col_indices = linear_sum_assignment(matrix_distance.T, maximize=False)
    #     total_distances = matrix_distance.T[row_indices, col_indices].sum()
    #     redundance_distances = [edit_distance(list1[i], [])[0] for i in range(len(list1)) if i not in col_indices]
    #     total_distances += sum(redundance_distances)
    # elif len(list1) < len(list2):
    #     row_indices, col_indices = linear_sum_assignment(matrix_distance, maximize=False)
    #     total_distances = matrix_distance[row_indices, col_indices].sum()
    #     redundance_distances = [edit_distance(list2[i], [])[0] for i in range(len(list2)) if i not in col_indices]
    #     total_distances += sum(redundance_distances)
    # else:
    #     row_indices, col_indices = linear_sum_assignment(matrix_distance, maximize=False)
    #     total_distances = matrix_distance[row_indices, col_indices].sum()
    row_indices, col_indices = linear_sum_assignment(matrix_distance, maximize=False)
    total_distances = matrix_distance[row_indices, col_indices].sum()
    return total_distances, 1 - total_distances / max(len(sum(list1, [])), len(sum(list2, [])))

def split_list_by_tag(input_list, tag="<p>"):  
    output_list = []  
    temp_list = []  
      
    for item in input_list:  
        if item == tag:  
            if temp_list:  
                output_list.append(temp_list)  
                temp_list = []  
        else:  
            temp_list.append(item)  
      
    if temp_list:  
        output_list.append(temp_list)  
      
    return output_list