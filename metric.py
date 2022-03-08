import os
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import coco
from detectron2.utils.logger import setup_logger
from optparse import OptionParser
import numpy as np
import cv2

class DetectRate():
    def __init__(self, trial_no, threshold, data_dir, img_dir):
        self.trial_no = trial_no
        self.threshold = threshold
        self.cfg = get_cfg()
        self.cfg.merge_from_file(f"./config/trial{self.trial_no}.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        self.test_dataset_dicts = coco.load_coco_json(data_dir + "test_dataset.json", img_dir + "test/", "my_dataset_test")
         
    def detectRate(self):
        predictor = DefaultPredictor(self.cfg)
        # init
        dataset_dicts = self.test_dataset_dicts
        detected_instance = 0
        gt_instance = 0
        false_instance =0
        gt_per_cls = np.zeros((22))
        detected_per_cls = np.zeros((22))
        for d in dataset_dicts:
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            pred_box = outputs['instances'].get_fields()['pred_boxes'].tensor
            pred_cls = outputs['instances'].get_fields()['pred_classes']
            pred_box_real = pred_box
            not_checked_box = pred_box.tolist()
            for i in range(len(d["annotations"])):
                true_cls = d["annotations"][i]['category_id']
                gt_X1, gt_Y1, gt_W, gt_H  = d["annotations"][i]['bbox']
                gt_instance += 1
                gt_per_cls[int(true_cls)] += 1
                update_detected = 0
                for j in range(len(pred_box)):###
                    p_X1, p_Y1, p_X2, p_Y2 = pred_box[j]###
                    xA = max(gt_X1, p_X1)
                    yA = max(gt_Y1, p_Y1)
                    xB = min(gt_X1+gt_W, p_X2)
                    yB = min(gt_Y1+gt_H, p_Y2)
                    interArea = max(0, xB-xA) * max(0, yB-yA)
                    if interArea>0:
                        # cls_check = pred_cls == true_cls
                        # if True in cls_check: ###
                        update_detected = 1
                        try:
                            not_checked_box.remove(pred_box[j].tolist())
                        except ValueError:
                            pass
                detected_instance += update_detected
                detected_per_cls[int(true_cls)] += update_detected
            false_instance += len(not_checked_box)
        detect_ratio = detected_instance/gt_instance
        false_ratio = false_instance/gt_instance
        detect_ratio_cls = [detected_per_cls[i]/gt_per_cls[i] for i in range(len(detected_per_cls))]
        print(f'threshold: {self.threshold}')
        print('='*20)
        print(f'gt_instance:        {gt_instance}')
        print(f'detected_instance:  {detected_instance}')
        print(f'detect_ratio:       {detect_ratio}')
        print(f'false_instance:     {false_instance}')
        print(f'false_ratio:        {false_ratio}')
        print('='*20)
        print(f'gt_per_cls:         {gt_per_cls}')
        print(f'detected_per_cls:   {detected_per_cls}')
        print(f'detect_ratio_cls:   {detect_ratio_cls}')
        print('='*20)
        return detect_ratio, false_ratio, detect_ratio_cls
        
    def bestmodel(self, train_trial):
        model_pth_list = [x for x in os.listdir(train_trial) if x.endswith('.pth') and x.startswith('model_')]
        model_pth_list.sort(reverse=True) # checking priority : latest checkpoints
        print(model_pth_list)
        detect_ratio_list = []
        false_ratio_list = []
        result_pth_list = []
        detect_ratio_cls_list = []     
        for model_pth in tqdm(model_pth_list[:int(len(model_pth_list)*0.35)], desc='model progress'):
            print('='*20)
            print(model_pth)
            result_pth_list.append(model_pth)
            self.cfg.MODEL.WEIGHTS = os.path.join(train_trial, model_pth)
            detect_ratio, false_ratio, detect_ratio_cls = self.detectRate()
            detect_ratio_list.append(detect_ratio)
            false_ratio_list.append(false_ratio)
            detect_ratio_cls_list.append(detect_ratio_cls)            
        for i in range(len(detect_ratio_list)):
            print(result_pth_list[i], ',', detect_ratio_list[i], ',', false_ratio_list[i])
        max_detect_ratio = max(detect_ratio_list)
        i_max = detect_ratio_list.index(max_detect_ratio)
        print('='*20)
        print(result_pth_list[i_max], ',', detect_ratio_list[i_max], ',', false_ratio_list[i_max])
        print(detect_ratio_cls_list[i_max])            

if __name__ == "__main__":
    setup_logger()
    parser = OptionParser()
    parser.add_option("--trial_no", dest="trial_no", default=0)
    parser.add_option("--gpu_no", dest="gpu_no", default=0)
    parser.add_option("--model_pth", dest="model_pth", default=False)  # ex) 0284999
    parser.add_option("--threshold", dest="threshold", default=0.3)
    
    (opts, args) = parser.parse_args()
    model_pth = 'model_' + str(opts.model_pth) + '.pth'
    
    data_dir = f'./data/annotations/' # label path
    img_dir = f'./data/images/' # image path    
    
    metric = DetectRate(opts.trial_no, opts.threshold, data_dir, img_dir)
    metric.detectRate()