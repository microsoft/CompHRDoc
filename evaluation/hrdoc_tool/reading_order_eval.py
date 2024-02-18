#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Author: Jiawei Wang (wangjiawei@mail.ustc.edu.cn)
# -----
# Copyright (c) 2024 Microsoft
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###

import os
import json
import tqdm
import glob
import logging
import argparse
import multiprocessing
from doc_utils import *
from doc_utils import transfer_tree_to_chain, edit_distance, split_list_by_tag, min_edit_distance_between_lists

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def assert_filetree(args):
    """ Make sure the file-tree of `gt_folder` and `pred_folder` are the same """
    gt_files = set([os.path.basename(x) for x in glob.glob(os.path.join(args.gt_folder, "*.json"))])
    pred_files = set([os.path.basename(x) for x in glob.glob(os.path.join(args.pred_folder, "*.json"))])
    if gt_files != pred_files:
        logging.error("ERROR while processing {}, ERR_CODE={}, message:{}".format(
            "filetree", 1, "pred-folder and gt-folder contains different json files"
        ))
        return 1
    else:
        return -1
    
def check_tree(json_file_path, gt_info, pred_info):
    if len(gt_info) != len(pred_info):
        logging.error("ERROR while processing {}, ERR_CODE={}, message:{}".format(
            json_file_path, 2, "number of nodes not equal"
        ))
        return 2
    parent_ids = {}
    for i in range(len(pred_info)):
        parent_ids[i] = pred_info[i]["parent_id"]
    for loop_time in range(len(pred_info)):
        Valid = True
        for item_id in range(len(pred_info)):
            if parent_ids[item_id] == -1: continue
            Valid = False
            parent_ids[item_id] = pred_info[parent_ids[item_id]]["parent_id"]
        if Valid: break
    if len(set(parent_ids.values())) != 1:
        vis_digraph_py(complete_json(pred_info, gt_info), os.path.splitext(json_file_path)[0])
        logging.error("ERROR while processing {}, ERR_CODE={}, message:{}".format(
            json_file_path, 3, "parent loop exists, visualization has been saved in {}".format(
                os.path.splitext(json_file_path)[0]
            )
        ))
        return 3
    return -1

def check_gt(gt_floating_reading_order_groups):
    for group in gt_floating_reading_order_groups:
    # 两个条件不符合GT,就不计算TEDS:
    #     1. 一个group中有多个figure/table
    #     2. 一个group里有多个caption开头
        figure = 0
        table = 0
        caption = 0
        for item in group:
            if item.startswith("figure"):
                figure += 1
            elif item.startswith("table"):
                table += 1
            elif item.lower().startswith("caption:fig") or item.lower().startswith("caption:tab"):
                caption += 1
        if figure > 1 or table > 1 or caption > 1:
            return False
    return True

def filtered_gt(gt_floating_reading_order_groups):
    filtered_gt_floating_reading_order_groups = []
    for group in gt_floating_reading_order_groups:
    # 两个条件不符合GT,就不计算TEDS:
    #     1. 一个group中有多个figure/table
    #     2. 一个group里有多个caption开头
        correct_group = []
        figure = []
        table = []
        caption = []
        all_caption = []
        for item in group:
            if item.startswith("figure"):
                figure.append(item)
            elif item.startswith("table"):
                table.append(item)
            elif item.lower().startswith("caption:fig") or item.lower().startswith("caption:tab"):
                caption.append(item)
                all_caption.append(item)
            else:
                all_caption.append(item)
        if len(caption) > 1 or len(figure) + len(table) > 1: # or len(all_caption) == 0: # len(all_caption) means label Table as Algorithm in HRDoc
            continue
        else:
            if len(figure) == 1:
                correct_group.append(figure[0])
                correct_group.extend(all_caption)
            elif len(table) == 1:
                correct_group.extend(all_caption)
                correct_group.append(table[0])
            else:
                continue
            filtered_gt_floating_reading_order_groups.append(correct_group)
    return filtered_gt_floating_reading_order_groups

def refine_pred(pred_floating_reading_order_groups):
    filtered_pred_floating_reading_order_groups = []
    for group in pred_floating_reading_order_groups:
        figure = []
        table = []
        caption = []
        for item in group:
            if item.startswith("figure"):
                figure.append(item)
            elif item.startswith("table"):
                table.append(item)
            else:
                caption.append(item)
        
        if len(figure) > 0:
            filtered_pred_floating_reading_order_groups.append([figure[0]] + caption)
            for f in figure[1:]:
                filtered_pred_floating_reading_order_groups.append([f])
        
        if len(table) > 0:
            filtered_pred_floating_reading_order_groups.append(caption + [table[0]])
            for t in table[1:]:
                filtered_pred_floating_reading_order_groups.append([t])
        
        if len(figure) == 0 and len(table) == 0:
            filtered_pred_floating_reading_order_groups.append(caption)
    return filtered_pred_floating_reading_order_groups
    
def _worker_cal_teds(result_queue, _tgt_jsons, args):
    total_info = {}
    # for json_file in tqdm.tqdm(_tgt_jsons):
    for json_file in _tgt_jsons:
        json_file_path = os.path.join(args.pred_folder, json_file)
        gt_info = json.load(open(os.path.join(args.gt_folder, json_file)))
        pred_info = json.load(open(json_file_path))
        gt_texts = [t['class']+":"+t['text'] for t in gt_info]
        gt_parent_idx = [t['parent_id'] for t in gt_info]
        gt_relation = [t['relation'] for t in gt_info]
        pred_texts = [t['class']+":"+t['text'] for t in pred_info]
        pred_parent_idx = [t['parent_id'] for t in pred_info]
        pred_relation = [t['relation'] for t in pred_info]
        # if check_tree(json_file_path, gt_info, pred_info) != -1: continue
        try:
            gt_tree = generate_doc_tree_from_log_line_level(gt_texts, gt_parent_idx, gt_relation)
            gt_reading_order_chain, gt_reading_order_chain_floating = transfer_tree_to_chain(gt_tree)
            pred_tree = generate_doc_tree_from_log_line_level(pred_texts, pred_parent_idx, pred_relation)
            pred_reading_order_chain, pred_reading_order_chain_floating = transfer_tree_to_chain(pred_tree)
            distance, teds = edit_distance(pred_reading_order_chain, gt_reading_order_chain)
            gt_floating_reading_order_groups = split_list_by_tag(gt_reading_order_chain_floating[1:])
            pred_floating_reading_order_groups = split_list_by_tag(pred_reading_order_chain_floating[1:])
            # if not check_gt(gt_floating_reading_order_groups):
            #     logging.error("ERROR while processing {}, ERR_CODE={}, message:{}".format(
            #         json_file_path, 5, "GT floating reading order not valid"
            #     ))
            #     continue
            # gt_floating_reading_order_groups = filtered_gt(gt_floating_reading_order_groups)
            # pred_floating_reading_order_groups = refine_pred(pred_floating_reading_order_groups)
            distance_floating, teds_floating = min_edit_distance_between_lists(gt_floating_reading_order_groups, pred_floating_reading_order_groups)
        except:
            vis_digraph_py(complete_json(pred_info, gt_info), os.path.splitext(json_file_path)[0])
            logging.error("ERROR while processing {}, ERR_CODE={}, message:{}".format(
                json_file_path, 4, "error when generate doc tree, visualization has been saved in {}".format(os.path.splitext(json_file_path)[0])
            ))
            continue
        teds_info = {
            "teds": teds, "distance": distance, "gt_nodes": len(gt_tree), "pred_nodes": len(pred_tree), "teds_floating": teds_floating, "distance_floating": distance_floating, "gt_floating_nodes": len(sum(gt_floating_reading_order_groups, [])), "pred_floating_nodes": len(sum(pred_floating_reading_order_groups, []))
        }
        total_info[json_file] = teds_info
    result_queue.put((total_info))

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--gt_folder", type=str, help="The folder storing ground-truth json files.")
    parser.add_argument("--pred_folder", type=str, help="The folder storing predicted json results.")
    parser.add_argument("--num_workers",type=int, default=0, help="The number of workers to process json files")
    
    args = parser.parse_args()
    logging.info("Args received, gt_folder: {}, pred_folder: {}".format(args.gt_folder, args.pred_folder))

    if assert_filetree(args=args) != -1: return
    logging.info("File tree matched, start parse through json files!")

    tgt_jsons = os.listdir(args.gt_folder)
    
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    if args.num_workers == 0:
        num_works = multiprocessing.cpu_count() // 2
    else:
        num_works = args.num_workers
    workers = list()
    for work_i in range(num_works):
        worker = multiprocessing.Process(
            target=_worker_cal_teds,
            args=(
                result_queue,
                tgt_jsons[work_i::num_works],
                args
            )
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    all_teds_info = {}
    for _ in range(num_works):
        _work_teds_info = result_queue.get()
        all_teds_info.update(_work_teds_info)

    teds_list = [v["teds"] for v in all_teds_info.values()]
    json.dump(all_teds_info, open(os.path.join(os.path.dirname(args.pred_folder), 'teds_info.json'), "w"), indent=4)
    distance_list = [v["distance"] for v in all_teds_info.values()]
    teds_floating_list = [v["teds_floating"] for v in all_teds_info.values()]
    distance_floating_list = [v["distance_floating"] for v in all_teds_info.values()]
    gt_nodes_list = [v["gt_nodes"] for v in all_teds_info.values()]
    pred_nodes_list = [v["pred_nodes"] for v in all_teds_info.values()]
    gt_floating_nodes_list = [v["gt_floating_nodes"] for v in all_teds_info.values()]
    pred_floating_nodes_list = [v["pred_floating_nodes"] for v in all_teds_info.values()]
    if len(teds_list):
        logging.info("macro_teds : {}".format(sum(teds_list)/len(teds_list)))
        logging.info("micro_teds : {}".format(1.0-float(sum(distance_list))/sum([max(gt_nodes_list[i], pred_nodes_list[i]) for i in range(len(teds_list))])))
        logging.info("macro_teds_floating : {}".format(sum(teds_floating_list)/len(teds_floating_list)))
        logging.info("micro_teds_floating : {}".format(1.0-float(sum(distance_floating_list))/sum([max(gt_floating_nodes_list[i], pred_floating_nodes_list[i]) for i in range(len(teds_floating_list))])))

if __name__ == "__main__":
    main()