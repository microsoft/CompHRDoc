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

import json
import os
import glob
import tqdm
import copy
import argparse
import logging
import os.path as osp

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARNING)
logger = logging.getLogger()
logging.getLogger().setLevel(logging.INFO)

def find_child_connect(ori_jd, line_id):
    ''' find the child and connect line of line_id'''
    child_ids = []
    connect_ids = []
    if ori_jd[line_id]["relation"] == "connect":
        return child_ids, connect_ids
    for line in ori_jd:
        if "visited" in line.keys(): continue
        if line["parent_id"] == line_id and line["relation"] == "contain":
            child_ids.append(line["line_id"])
        elif line["parent_id"] in child_ids and line["relation"] == "equality":
            child_ids.append(line["line_id"])
        elif line["parent_id"] == line_id and line["relation"] == "connect":
            connect_ids.append(line["line_id"])
        elif line["parent_id"] in connect_ids and line["relation"] == "connect":
            connect_ids.append(line["line_id"])
    return child_ids, connect_ids

class PaperNode(object):
    def __init__(self, ori_jd, line_id, label) -> None:
        self.text = ori_jd[line_id]["text"]
        self.box = ori_jd[line_id]["box"]
        self.line_id = line_id
        self.label = label
        self.child = list() # List of PaperNode
        self.connect = list() # List of PaperNode
        self.get_content(ori_jd)
    def get_content(self, ori_jd):
        child_ids, connect_ids = find_child_connect(ori_jd, self.line_id)
        if len(child_ids) == 0 and len(connect_ids) == 0:
            return
        else:
            if len(child_ids):
                for child_id in child_ids:
                    class_ = ori_jd[child_id]["class"]
                    label = class_ if "section" in class_ else "paraline"
                    self.child.append(PaperNode(ori_jd, child_id, label))
                    ori_jd[child_id]["visited"] = True
            if len(connect_ids):
                for connect_id in connect_ids:
                    self.connect.append(PaperNode(ori_jd, connect_id, "opara"))
                    ori_jd[connect_id]["visited"] = True
    def self_text(self):
        if len(self.connect) == 0:
            return self.text
        else:
            cur_text = self.text
            for connect_node in self.connect:
                # When this line's last word is splited into two subword with the second subword in the next line
                if cur_text and cur_text[-1] == "-": 
                    cur_text = cur_text[:-1] + connect_node.self_text()
                else: 
                    cur_text = cur_text[:] + " " + connect_node.self_text()
            return cur_text
    def gather_info(self):
        gathered_info = {"label": self.label, "text": self.self_text()}
        if len(self.child) == 0:
            return gathered_info
        else:
            child_info = []
            for child_node in self.child:
                child_info.append(child_node.gather_info())
            gathered_info["child"] = child_info
        return gathered_info

def check_all_visited(pdf_path, ori_jd):
    not_visited = []
    for line in ori_jd:
        if "visited" not in line.keys(): 
            not_visited.append(line["line_id"])
    if len(not_visited):
        logger.warning("File {} has nodes not visited! {}".format(pdf_path, repr(not_visited)))

def transfer_json(pdf_path, ori_jd):
    tgt_json = {"title": [], "author": [], "affili": [], "mail": [], "section": [], "figure": [], "table": [], "footer": [], "footnote": []}
    meta = ["title", "author", "affili", "mail", "footer", "footnote"]
    fig_tab = ["figure", "table"]
    hierarchy = ["section"]
    # * Handling meta node
    for line in ori_jd:
        if "visited" in line.keys(): continue
        if line["class"] in meta:
            tgt_json[line["class"]].append({"text": line["text"], "box": line["box"], "line_id": line["line_id"], "page": line["page"]})
            line["visited"] = True
    for key in meta:
        for item in tgt_json[key]:
            child_ids, connect_ids = find_child_connect(ori_jd, item["line_id"])
            # * Meta node should has no child node, may have connected node
            assert len(child_ids) == 0, "Find meta has child in {}, line id: {}".format(pdf_path, item["line_id"])
            if len(connect_ids) != 0:
                item["connect"] = []
                for connect_id in connect_ids:
                    line = ori_jd[connect_id]
                    item["connect"].append({"text": line["text"], "box": line["box"], "line_id": line["line_id"], "page": line["page"]})
                    line["visited"] = True
    # * Renew fig/tab
    for line in ori_jd:
        if "visited" in line.keys(): continue
        if line['text'] == "Figure" and line["box"][3] - line['box'][1] > 8:
            line['class'] = 'fig'
        if line['text'] == "Table" and line["box"][3] - line['box'][1] > 8:
            line['class'] = 'tab'
    
    # * Handling fig/tab node
    for line in ori_jd: # find all img/tab/alg position
        if "visited" in line.keys(): continue
        if line["class"] in fig_tab:
            tgt_json[line["class"]].append({"text": line["text"], "box": line["box"], "line_id": line["line_id"], "page": line["page"]})
            line["visited"] = True
    for key in fig_tab: # find all caption
        for item in tgt_json[key]:
            child_ids, connect_ids = find_child_connect(ori_jd, item["line_id"])
            # * fig_tab node should has no connect node, may have child node
            assert len(connect_ids) == 0, "Find {} has connent node in {}, line id: {}".format(key, pdf_path, item["line_id"])
            if len(child_ids) != 0:
                item["caption"] = []
                for child_id in child_ids:
                    line = ori_jd[child_id]
                    item["caption"].append({"text": line["text"], "box": line["box"], "line_id": line["line_id"], "page": line["page"]})
                    line["visited"] = True
                    cur_caption = item["caption"][-1]
                    sub_child_ids, sub_connect_ids = find_child_connect(ori_jd, child_id)
                    assert len(sub_child_ids) == 0, "Find {}'s caption has child in {}, line id: {}".format(key, pdf_path, item["line_id"])
                    if len(sub_connect_ids) != 0:
                        cur_caption["connect"] = []
                        for sub_connect_id in sub_connect_ids:
                            line = ori_jd[sub_connect_id]
                            cur_caption["connect"].append({"text": line["text"], "box": line["box"], "line_id": line["line_id"], "page": line["page"]})
                            line["visited"] = True
    # Renew fstline
    prev_fstline = -1
    for line_id in range(len(ori_jd)):
        if "visited" in ori_jd[line_id].keys(): continue
        if ori_jd[line_id]['class'] == 'fstline':
            if prev_fstline > 0:
                ori_jd[line_id]['parent_id'] = prev_fstline
                ori_jd[line_id]['relation'] = 'equality'
            prev_fstline = line_id
        if ori_jd[line_id]['class'] == "section":
            prev_fstline = -1
    # * Handling section/para node
    for line in ori_jd: # find all img/tab/alg position
        if "visited" in line.keys(): continue
        if line["class"] in hierarchy:
            tgt_json[line["class"]].append(PaperNode(ori_jd, line["line_id"], "section"))
            line["visited"] = True
    # check_all_visited(pdf_path, ori_jd)
    return tgt_json

def gather_connect_text(only_connect_node):
    cur_text = only_connect_node["text"]
    if "connect" in only_connect_node.keys():
        for connect in only_connect_node["connect"]:
            if cur_text and cur_text[-1] == "-":
                cur_text = cur_text[:-1] + connect["text"]
            else:
                cur_text = cur_text + " " + connect["text"]
    return cur_text

def gather_text_info(file_path, tgt_json):
    final_json = copy.deepcopy(tgt_json)
    meta = ["title", "author", "affili", "mail", "footer", "footnote"]
    fig_tab = ["figure", "table"]
    hierarchy = ["section"]
    for key in meta:
        text_list = []
        for item in final_json[key]:
            text_list.append(gather_connect_text(item))
        final_json[key] = text_list
    for key in fig_tab:
        fig_tab_caption_list = []
        for item in final_json[key]:
            box_cap = {"box": item["box"], "page": item["page"]}
            if "caption" in item.keys():
                assert len(item["caption"]) == 1, "Find {} has more than one caption in {}, line id: {}".format(key, file_path, item["line_id"])
                caption_text = gather_connect_text(item["caption"][0])
                box_cap["caption"] = caption_text
            fig_tab_caption_list.append(box_cap)
        final_json[key] = fig_tab_caption_list
    for key in hierarchy:
        hier_info = []
        for item in final_json[key]:
            assert isinstance(item, PaperNode)
            hier_info.append(item.gather_info())
        final_json[key] = hier_info
    return final_json
    
def get_json_paths(folder, recursive=False):
    if not osp.exists(folder):
        return "Error: No such file or directory: {}".format(folder)
    if recursive:
        json_list = glob.glob(osp.join(folder, "**/*.json"), recursive=True)
    else:
        json_list = glob.glob(osp.join(folder, "*.json"), recursive=False)
    return json_list

def parser_args():
    argparser = argparse.ArgumentParser(description="Transfer json that contain information for each line to json that hierarchically store document information")
    argparser.add_argument("--src_json_path", default="strcut_recover/experiment/pred_result.json")
    argparser.add_argument("--save_folder", default="generated_json")
    args = argparser.parse_args()
    return args

def main():
    args = parser_args()
    json_files = json.load(open(args.src_json_path, "r"))
    pbar = tqdm.tqdm(json_files)
    os.makedirs(args.save_folder, exist_ok=True)
    for one_pdf_data in pbar:
        try:
            pdf_path = one_pdf_data["pdf"]
            jdata = one_pdf_data["decoded_json"]
            bname = osp.basename(pdf_path)[:-4] + ".json" \
                if osp.basename(pdf_path).endswith(".pdf") else osp.basename(pdf_path) + ".json"
            pbar.set_description("Processing {}".format(bname))
            final_json_file = osp.join(args.save_folder, bname)
            # if osp.exists(final_json_file): continue
            tgt_json = transfer_json(pdf_path, jdata)
            final_json = gather_text_info(pdf_path, tgt_json)
            json.dump(final_json, open(final_json_file, "w"), indent=4)
        except BaseException as e:
            logger.info("Meet error when dealing {}".format(one_pdf_data["pdf"]))
            logger.info("Error message: {}".format(e))
            continue

if __name__ == "__main__":
    main()
