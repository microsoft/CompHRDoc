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
import json
import tqdm
import glob
import logging
import argparse
import multiprocessing
from doc_utils import *

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
    
def _worker_cal_teds(result_queue, _tgt_jsons, args):
    total_info = {}
    for json_file in tqdm.tqdm(_tgt_jsons):
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
            pred_tree = generate_doc_tree_from_log_line_level(pred_texts, pred_parent_idx, pred_relation)
            distance, teds = tree_edit_distance(pred_tree, gt_tree)
        except:
            vis_digraph_py(complete_json(pred_info, gt_info), os.path.splitext(json_file_path)[0])
            logging.error("ERROR while processing {}, ERR_CODE={}, message:{}".format(
                json_file_path, 4, "error when generate doc tree, visualization has been saved in {}".format(os.path.splitext(json_file_path)[0])
            ))
            continue
        teds_info = {
            "teds": teds, "distance": distance, "gt_nodes": len(gt_tree), "pred_nodes": len(pred_tree)
        }
        total_info[json_file] = teds_info
    result_queue.put((total_info))

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--gt_folder", type=str, help="The folder storing ground-truth json files.")
    parser.add_argument("--pred_folder", type=str, help="The folder storing predicted json results.")
    
    args = parser.parse_args()
    logging.info("Args received, gt_folder: {}, pred_folder: {}".format(args.gt_folder, args.pred_folder))

    if assert_filetree(args=args) != -1: return
    logging.info("File tree matched, start parse through json files!")

    tgt_jsons = os.listdir(args.gt_folder)
    
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    num_works = multiprocessing.cpu_count() // 2
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
    gt_nodes_list = [v["gt_nodes"] for v in all_teds_info.values()]
    pred_nodes_list = [v["pred_nodes"] for v in all_teds_info.values()]
    if len(teds_list):
        logging.info("macro_teds : {}".format(sum(teds_list)/len(teds_list)))
        logging.info("micro_teds : {}".format(1.0-float(sum(distance_list))/sum([max(gt_nodes_list[i], pred_nodes_list[i]) for i in range(len(teds_list))])))

if __name__ == "__main__":
    main()