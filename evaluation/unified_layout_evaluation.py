import os
import json
import numpy as np
from tqdm import tqdm
import sys

from detectron2.evaluation import DatasetEvaluator
from detectron2.data import MetadataCatalog

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class UniLayoutEvaluator(DatasetEvaluator):
    """Evaluator for Unified Layout Analysis.

    Args:
        DatasetEvaluator (_type_): _description_
    """
    
    def __init__(self, dataset_name, output_dir, as_two_stage=False) -> None:
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "COMP_HRDOC_HR_TEST"
        """
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self.as_two_stage = as_two_stage
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
    
    def reset(self) -> None:
        """Reset the internal state to prepare for a new round of evaluation.
        """
        pass

    def process(self, inputs, outputs) -> None:
        """Process an image and its annotations to evaluate.
        
        Args:
            inputs (dict): a dict that contains the input image
            outputs (dict): a dict that contains the output of the model
        """
        if os.path.exists(os.path.join(self._output_dir, 'hr_json')) == False:
            os.mkdir(os.path.join(self._output_dir, 'hr_json'))
        hr_path = os.path.join(self._output_dir, 'hr_json', inputs[0]['file_name'][0].split('/')[-1].split('_')[0] + ".json")
        with open(hr_path, "w") as json_file:
            json.dump(outputs['output_for_document'], json_file, indent=4, cls=MyEncoder)

        # if self.as_two_stage:
        if os.path.exists(os.path.join(self._output_dir, 'det_json')) == False:
            os.mkdir(os.path.join(self._output_dir, 'det_json'))  
    
        for i, page_level_det in enumerate(outputs['output_for_page']):
            if len(page_level_det) == 0:
                page_det_path = os.path.join(self._output_dir, 'det_json', outputs['output_for_page'][0][0]['file_name'].split('_')[0] + f'_{i}' + ".json")
            else:
                page_det_path = os.path.join(self._output_dir, 'det_json', page_level_det[0]['file_name'][:-4] + ".json")
            with open(page_det_path, "w") as json_file:
                json.dump(page_level_det, json_file, indent=4, cls=MyEncoder)
                
        """
        Table of Contents Extraction
        """
        toc_path = os.path.join(self._output_dir, 'toc_json', inputs[0]['file_name'][0].split('/')[-1].split('_')[0] + ".json")
        with open(toc_path, "w") as json_file:
            json.dump(outputs['output_for_toc'], json_file, indent=4, cls=MyEncoder)
            
        """
        Hierarchical Document Structure Reconstruction
        """
        hr_path = os.path.join(self._output_dir, 'hr_json', inputs[0]['file_name'][0].split('/')[-1].split('_')[0] + ".json")
        with open(hr_path, "w") as json_file:
            json.dump(outputs['output_for_document'], json_file, indent=4, cls=MyEncoder)
        
        return
    
    def evaluate(self) -> None:
        """Evaluate the predictions collected so far.
        """
        
        gt_folder = "datasets/Comp-HRDoc/HRDH_MSRA_POD_TEST/test_eval"
        pred_folder = os.path.join(self._output_dir, 'hr_json')
        save_folder = os.path.join(self._output_dir, 'logical_role_json')
        if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)
        print('Reading Order Prediction Evaluation on Comp-HRDoc')
        print('********************************************************')
        cmd = f"{sys.executable} projects/unified_layout_analysis/evaluation/hrdoc_tool/reading_order_eval.py --gt_folder {gt_folder} --pred_folder {pred_folder} --num_workers 8 \n"
        os.system(cmd)
        
        if not self.as_two_stage:
            gt_jsons = sorted(os.listdir(gt_folder))
            for gt_json_file in tqdm(gt_jsons):
                gt_json = json.load(open(os.path.join(gt_folder, gt_json_file)))
                pred_json_file = gt_json_file
                pred_json = json.load(open(os.path.join(pred_folder, pred_json_file)))
                rearranged_pred = rearrange_pred(pred_json, gt_json)
                with open(os.path.join(save_folder, gt_json_file), 'w') as json_file:
                    json.dump(rearranged_pred, json_file, indent=4)
                    
            print('Logical Role Classification Evaluation on Comp-HRDoc')
            print('********************************************************')
            cmd = f"{sys.executable} projects/unified_layout_analysis/evaluation/hrdoc_tool/classify_eval.py --gt_folder {gt_folder} --pred_folder {save_folder} \n"
            os.system(cmd)
        else:
            print('Page Object Detection Evaluation on Comp-HRDoc')
            print('********************************************************')
            pod_gt = "datasets/Comp-HRDoc/HRDH_MSRA_POD_TEST/coco_test.json"
            pod_pred = os.path.join(self._output_dir, 'det_json')
            cmd = f"{sys.executable} projects/unified_layout_analysis/evaluation/hrdoc_tool/page_object_detection_eval.py --gt_anno {pod_gt} --pred_folder {pod_pred}\n"
            os.system(cmd)
            
        print('Table of Contents Extraction Evaluation on Comp-HRDoc')
        print('********************************************************')
        toc_gt = "datasets/Comp-HRDoc/HRDH_MSRA_POD_TEST/test_eval_toc/"
        toc_pred = os.path.join(self._output_dir, 'toc_json')
        cmd = f"{sys.executable} projects/unified_layout_analysis/evaluation/hrdoc_tool/teds_eval.py --gt_anno {toc_gt} --pred_folder {toc_pred}\n"
        os.system(cmd)
        
        print('Hierarchical Document Structure Reconstruction Evaluation on Comp-HRDoc')
        print('********************************************************')
        hds_gt = "datasets/Comp-HRDoc/HRDH_MSRA_POD_TEST/test_eval/"
        hds_pred = os.path.join(self._output_dir, 'hr_json')
        cmd = f"{sys.executable} projects/unified_layout_analysis/evaluation/hrdoc_tool/teds_eval.py --gt_anno {hds_gt} --pred_folder {hds_pred}\n"
        
        return


def rearrange_pred(pred_json, gt_json):
    pred_box = [pred_json[i]['box'] for i in range(len(pred_json))]
    gt_box = [gt_json[i]['box'] for i in range(len(gt_json))]
    assert len(pred_box) == len(gt_box)
    rearranged_pred = []
    for i, box in enumerate(gt_box):
        try:
            found = False
            for index, p_b in enumerate(pred_box):
                if p_b == box and pred_json[index]['page'] == gt_json[i]['page']:
                    found = True
                    break
            if not found:
                print("Box not found in pred_json: {}, Class: {}".format(box, gt_json[i]['class']))
                rearranged_pred.append(gt_json[i])
                continue
            rearranged_pred.append(pred_json[index])
        except:
            print("Text not found in pred_json: {}".format(box))
            rearranged_pred.append(gt_json[i])
            
    return rearranged_pred