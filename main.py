import os
import cv2
from optparse import OptionParser
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from training import train
from inference import Inference
from metric import DetectRate

def main():
    setup_logger()
    parser = OptionParser()
    parser.add_option("--mode", dest='mode', default='train')
    parser.add_option("--trial_no", dest="trial_no", default=0)
    parser.add_option("--gpu_no", dest="gpu_no", default=0)
    parser.add_option("--model_pth", dest="model_pth", default=False)  # ex) 0284999
    parser.add_option("--ims_per_batch", dest="ims_per_batch", default=20)
    parser.add_option("--max_iter", dest="max_iter", default=500000)
    parser.add_option("--batch_size_per_image", dest="batch_size_per_image", default=512)
    parser.add_option("--base_lr", dest="base_lr", default=0.0001)
    parser.add_option("--load_ox", dest="load_ox", default=False)
    parser.add_option('--eval_period', dest='eval_period', default=5000)
    parser.add_option('--weight_decay', dest='weight_decay', default=0)
    parser.add_option('--multiple', dest='multiple', default=10) # number of random crop
    parser.add_option("--threshold", dest="threshold", default=0.3)
    parser.add_option("--config", dest="config", default='./config/steel.yaml')
    parser.add_option("--backbone", dest="backbone", default='ResNext-101')

    # set type of parsers
    (opts, args) = parser.parse_args()
    model_pth = 'model_' + str(opts.model_pth) + '.pth'
    
    # set path
    data_dir = f'./data/annotations/' # label path
    img_dir = f'./data/images/' # image path
    output_dir = f'./saved_model/{opts.mode}_trial{opts.trial_no}/'
    output_img_dir = f'./saved_prediction/{opts.mode}_trial{opts.trial_no}/'
    if opts.backbone == 'ResNext-101':
        backbone = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
    elif opts.backbone == 'ResNet-101':
        backbone = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
    elif opts.backbone == 'ResNet-50':
        backbone = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
    using_model = backbone

    # use saved checkpoint model if model_pth parser exists
    train_trial = f"./saved_model/train_trial{opts.trial_no}/"
    if bool(opts.model_pth):
        output_dir = f'./saved_model/train_trial{opts.trial_no}_continue/'
        output_img_dir = f'./saved_prediction/train_trial{opts.trial_no}_continue/'
        using_model = train_trial + model_pth

    # set gpu device and make output directory
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu_no)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)

    # set config
    cfg = get_cfg()
    cfg.merge_from_file(opts.config)
    cfg.TEST.EVAL_PERIOD = int(opts.eval_period)
    cfg.SOLVER.CHECKPOINT_PERIOD = int(opts.eval_period)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(using_model) if using_model == backbone else using_model
    cfg.SOLVER.IMS_PER_BATCH = int(opts.ims_per_batch)
    cfg.SOLVER.BASE_LR = float(opts.base_lr)
    cfg.SOLVER.MAX_ITER = int(opts.max_iter)
    cfg.SOLVER.WEIGHT_DECAY = float(opts.weight_decay)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = int(opts.batch_size_per_image) #default is 512
    cfg.OUTPUT_DIR = output_dir
    
    # save config
    with open(f"./config/trial{opts.trial_no}.yaml", "w") as f:
        f.write(cfg.dump())
    
    if opts.mode == "train":
        train(cfg, data_dir, img_dir, opts.multiple, output_img_dir, load_ox=opts.load_ox, vis=False)
        metric = DetectRate(opts.trial_no, opts.threshold, data_dir, img_dir)
        metric.bestmodel(train_trial)
    elif opts.mode == "inference":
        inference = Inference(cfg, data_dir, img_dir, opts.multiple, output_img_dir, opts.threshold, vis=False)
        inference.inference()
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
