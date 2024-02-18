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
import itertools

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--gt_anno", type=str, help="The annotation storing ground-truth (coco format).")
    parser.add_argument("--pred_folder", type=str, help="The folder storing predicted json results.")
    
    args = parser.parse_args()
    logging.info("Args received, gt_folder: {}, pred_folder: {}".format(args.gt_anno, args.pred_folder))
    
    cocoGt = COCO(args.gt_anno)
    image_ids = cocoGt.getImgIds()
    image_ids.sort()
    num_images = len(image_ids)
    logging.info("total doc images number {}".format(num_images))

    ids2names = {}
    for image in cocoGt.dataset["images"]:
        ids2names[image['id']] = image['file_name']
        
    predictions = []
    
    for i, image_id in enumerate(image_ids):
        image_name = ids2names[image_id]
        det_results = json.load(open(os.path.join(args.pred_folder, image_name[:-4] + ".json")))
        for a in cocoGt.imgs.values():
            if a['file_name'] == image_name:
                for det_result in det_results:
                    det_result['image_id'] = a['id']
        
        predictions.append(det_results)
        
    coco_results = list(itertools.chain(*[x for x in predictions]))
    cocoDt = cocoGt.loadRes(coco_results)
    # running customized coco evaluation
    coco_eval = COCOeval(cocoGt, cocoDt, "segm")
    coco_eval.params.imgIds  = list(image_ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # 输出每个类别在IoU 0.5:0.95范围内的mAP  
    category_aps = {}
    category_aps_iou_05 = {}
    for category_id in cocoGt.getCatIds():  
        # 获取类别名称  
        category_name = cocoGt.loadCats(category_id)[0]["name"]  
        
        # 获取该类别在IoU 0.5:0.95范围内的AP  
        category_ap = coco_eval.eval["precision"][:, :, category_id - 1, 0, -1].mean()  
        
        # 获取该类别在IoU 0.5的AP  
        category_ap_iou_05 = coco_eval.eval["precision"][0, :, category_id - 1, 0, -1].mean()  
        
        # 保存到字典中    
        category_aps[category_name] = category_ap    
        category_aps_iou_05[category_name] = category_ap_iou_05  
        
    # 打印每个类别的mAP    
    for category_name, ap in category_aps.items():    
        print(f"AP for {category_name} (IoU 0.5:0.95): {ap:.4f}")  
    
    # 打印每个类别在IoU 0.5的AP  
    for category_name, ap in category_aps_iou_05.items():    
        print(f"AP for {category_name} (IoU 0.5): {ap:.4f}")

if __name__ == "__main__":
    main()