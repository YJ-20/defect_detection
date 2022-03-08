import os
import cv2
from optparse import OptionParser
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import coco
import pandas as pd
from data import load_data

class Inference():
    def __init__(self, cfg, data_dir, img_dir, multiple, output_img_dir, threshold, vis=False):
        self.cfg = cfg
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.multiple = multiple
        self.output_img_dir = output_img_dir
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold   # set the testing threshold for this model
        self.vis = vis
    
    def evaluate(self):
        evaluator = COCOEvaluator("my_dataset_test", self.cfg, False, output_dir=self.cfg.OUTPUT_DIR)
        data_loader = build_detection_test_loader(self.cfg, "my_dataset_test")
        inference_on_dataset(self.predictor.model, data_loader, evaluator)
        
    def saveoutput(self, test_metadata):
        columns = ['image_id', 'X1', 'Y1', 'X2', 'Y2', 'category_id', 'score']
        df = pd.DataFrame(columns=columns)
        dataset_dicts = coco.load_coco_json(self.data_dir + "test_dataset.json", self.img_dir + "test/", "my_dataset_test")
        for d in dataset_dicts:
            im = cv2.imread(d["file_name"])
            outputs = self.predictor(im)
            image_id = d['file_name'].split('/')[-1]
            boxes = outputs['instances'].pred_boxes.tensor.to('cpu').numpy()
            pred_classes = outputs['instances'].pred_classes.to('cpu').numpy()
            scores = outputs['instances'].scores.to('cpu').numpy()
            i=0
            for item in boxes:
                x1, y1, x2, y2 = item
                dl = pd.Series([image_id, x1, y1, x2, y2, pred_classes[i], scores[i]], columns)
                df = df.append(dl,ignore_index=True)
                i+=1
            if self.vis:
                v = Visualizer(im[:, :, ::-1], metadata=test_metadata)
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                cv2.imwrite(self.output_img_dir + d["file_name"].split('/')[-1],v.get_image()[:, :, ::-1])
            print("saving test image: {}".format(d["file_name"]))    
        df.to_csv(self.output_img_dir + f'inference_trial{self.trial_no}_{self.threshold}.csv', index=False)
        
    def inference(self):
        mode = 'inference'
        test_metadata = load_data(mode, self.data_dir, self.img_dir, self.multiple)
        self.predictor = DefaultPredictor(self.cfg)
        self.evaluate()
        self.saveoutput(test_metadata)
        
