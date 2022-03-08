from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import coco

def load_data(mode, data_dir, img_dir, multiple):
    if mode == 'train':
        # register data
        train_dataset_dicts = coco.load_coco_json(data_dir + f"train_aug{multiple}_dataset.json", img_dir + f"train_aug{multiple}/", "my_dataset_train")
        DatasetCatalog.register("my_dataset_train", lambda : train_dataset_dicts)
        print('register done!')
        
        validation_dataset_dicts = coco.load_coco_json(data_dir + "val_dataset.json", img_dir + "val/", "my_dataset_validation")
        DatasetCatalog.register("my_dataset_validation", lambda : validation_dataset_dicts)
        print('register done!')
        
        test_dataset_dicts = coco.load_coco_json(data_dir + "test_dataset.json", img_dir + "test/", "my_dataset_test")
        DatasetCatalog.register("my_dataset_test", lambda : test_dataset_dicts)
        print('register done!')
        
        train_metadata = MetadataCatalog.get("my_dataset_train")
        validation_metadata = MetadataCatalog.get("my_dataset_validation")
        test_metadata = MetadataCatalog.get("my_dataset_test")
        return train_metadata, validation_metadata, test_metadata
    elif mode == 'inference':
        test_dataset_dicts = coco.load_coco_json(data_dir + "test_dataset.json", img_dir + "test/", "my_dataset_test")
        DatasetCatalog.register("my_dataset_test", lambda : test_dataset_dicts)
        MetadataCatalog.get("my_dataset_test").thing_classes = [
                'Ps_StDent_Single', 'Ps_DullMark', 'BlackLine', 'DirtyScab', 'LineScab', 'Scrape', 'Machalhum', 
                'Dent', 'Scratch', 'PinchTree', 'OilDrop', 'Dirty', 'EdgeCrack', 'Hole', 'WeldHole', 
                'WeldLine', 'Scale', 'ReOxidation_Line', 'PitScale', 'WetDrop', 'Ps_SurfaceEtc', 'EdgeBending'
            ]
        test_metadata = MetadataCatalog.get("my_dataset_test")
        return test_metadata
    
    