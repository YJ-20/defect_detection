import os
import shutil
import pandas as pd
import json
import random
from collections import OrderedDict
from tqdm import tqdm
import cv2
from optparse import OptionParser

class Dataset():
    def __init__(self):
        csv_data_na = pd.read_csv('./MSoS_main_add.csv')  # changeable
        csv_data = csv_data_na.copy()
        csv_data.dropna(inplace=True)
        csv_data.drop(csv_data.loc[csv_data['Field']==0].index, inplace=True)    # darkness field drop
        self.csv_data = csv_data
        self.data_dir = './data/'   # changeable
        self.dest_dir = self.data_dir + 'images/'
        self.train_dir = self.dest_dir + 'train/'
        self.val_dir = self.dest_dir + 'val/'
        self.test_dir = self.dest_dir + 'test/'
        self.annotation_dir = self.data_dir + 'annotations/'
        
    def small(self, category_id_list):
        small_set = set()
        train_small = []
        val_small = []
        test_small = []
        for id in category_id_list:
            df = self.csv_data.loc[self.csv_data['category_id']==id]
            df_set = set(df['Filename'].values)
            small_set = small_set.union(df_set)
            df_list = list(df_set)
            df_len = len(df_list)
            train_num = round(df_len * 0.7)
            val_num = round(df_len * 0.1)
            train_small = train_small + df_list[:train_num]
            val_small = val_small + df_list[train_num:train_num+val_num]
            test_small = test_small + df_list[train_num+val_num:]
        return small_set, train_small, val_small, test_small

    def annotation(self, phase, phase_image, phase_dir, annotation_dir):
        full_data = OrderedDict()
        full_data['info'] = {}
        full_data['licenses'] = []
        full_data['images'] = []
        full_data['annotations'] = []
        full_data['categories'] = []

        i = 0
        count = 0
        # for images
        for img in phase_image:
            image = OrderedDict() # dict per image
            full_img_path = phase_dir + img # for file_name
            image['file_name'] = full_img_path
            image['height'] = int(1000)
            image['width'] = int(2048)
            image['id'] = int(i)                      
            
            full_data['images'].append(image)
            
            img_data = self.csv_data.loc[self.csv_data['Filename'] == img]   # small dataframe
            category_list = list(img_data['category_id'].values)    
            
            for j in range(len(img_data)):
                x0, y0, x1, y1 = img_data.loc[:,['X0','Y0', 'X1', 'Y1']].iloc[j,:]
                w, h = x1 - x0, y1 - y0
                area = float(w) * float(h)
                
                annotations = OrderedDict()
                annotations['segmentation'] = []
                annotations['area'] = float(area)
                annotations['iscrowd'] = int(0)         
                annotations['image_id'] = int(i)
                annotations['bbox'] = [float(x0), float(y0), float(w), float(h)]
                annotations['category_id'] = int(category_list[j] - 1)
                annotations['id'] = count
                count += 1
                full_data['annotations'].append(annotations)
            i += 1
            print(f'image count : {i} / {len(phase_image)}, object count : {count}')
            
            
        category_list = [
            'Ps_StDent_Single', 'Ps_DullMark', 'BlackLine', 'DirtyScab', 'LineScab', 'Scrape', 'Machalhum', 
            'Dent', 'Scratch', 'PinchTree', 'OilDrop', 'Dirty', 'EdgeCrack', 'Hole', 'WeldHole', 
            'WeldLine', 'Scale', 'ReOxidation_Line', 'PitScale', 'WetDrop', 'Ps_SurfaceEtc', 'EdgeBending'
        ]
        for k in range(22):
            category = OrderedDict()
            category['supercategory'] = category_list[k]
            category['id'] = int(k)
            category['name'] = category_list[k]
            
            full_data['categories'].append(category)
                        
        with open(annotation_dir + f'{phase}_dataset.json', 'w', encoding = 'utf-8') as make_file: 
            json.dump(full_data, make_file)

    def split_annot(self):
        img_set = set(self.csv_data['Filename'].values)
        
        # equal distribution split for small categories
        category_id_list = [3, 6, 12, 13, 18, 20]    # dirtyscab, Machalhum, dgecrack, hole, pitscale, Ps_SurfaceEtc
        small_set, train_small, val_small, test_small = self.small(category_id_list)

        img_part_set = img_set - small_set
        img_part_list = list(img_part_set)
        img_part_len = len(img_part_list)

        train_num = round(img_part_len * 0.7)
        val_num = round(img_part_len * 0.1)

        train_image = img_part_list[:train_num] + train_small
        val_image = img_part_list[train_num:train_num+val_num] + val_small
        test_image = img_part_list[train_num+val_num:] + test_small

        random.shuffle(train_image)
        random.shuffle(val_image)
        random.shuffle(test_image)
   

        # dir
        for d in [self.data_dir, self.dest_dir, self.train_dir, self.val_dir, self.test_dir, self.annotation_dir]:
            if not os.path.exists(d):
                os.mkdir(d)
        
        # copy images
        src = '../SSDD_DeepTool/SSDD_MSOS/Images/' # changeable
        for phase, phase_image, phase_dir in [('train', train_image, self.train_dir), ('val', val_image, self.val_dir), ('test', test_image, self.test_dir)]:
            for img in tqdm(phase_image, desc=f'{phase}_copy'):
                shutil.copy(src + img, phase_dir + img)
            self.annotation(phase, phase_image, phase_dir, self.annotation_dir)

    ##### for augmentation #####
    def flip_GT(self, o_img, axis, X0, Y0, X1, Y1):
        origin_y = o_img.shape[0]
        origin_x = o_img.shape[1]
        
        if axis == 0:
            new_X0 = origin_x
            new_Y0 = origin_y - Y0
            new_X1 = origin_x
            new_Y1 = origin_y - Y1
        if axis == 1:
            new_X0 = origin_x - X0
            new_Y0 = origin_y
            new_X1 = origin_x - X1
            new_Y1 = origin_y
        if axis == -1:
            new_X0 = origin_x - X0
            new_Y0 = origin_y - Y0
            new_X1 = origin_x - X1
            new_Y1 = origin_y - Y1
        return new_X0, new_Y0, new_X1, new_Y1

    def rescale_GT(self, o_img,  X0, Y0, X1, Y1, target_size_x, target_size_y): # file_name : "P_000000.jpg" / coordinates (x1, y1, x2, y2)
        origin_y = o_img.shape[0]
        origin_x = o_img.shape[1]
        
        x_scale = target_size_x / origin_x
        y_scale = target_size_y / origin_y
        
        new_x0 = X0 * x_scale
        new_y0 = Y0 * y_scale
        new_x1 = X1 * x_scale
        new_y1 = Y1 * y_scale
        return new_x0, new_y0, new_x1, new_y1
 
    def crop_coords(self, multiple):
        # crop point set
        crop_range = 100
        aug_multiple = multiple
        r_coords = []
        count = 0
        random.seed(100)
        for i in range(10000):
            rx = random.randint(1, crop_range)
            ry = random.randint(1, crop_range)
            if (rx, ry) not in r_coords:
                count += 1
                r_coords.append((rx, ry))
            else:
                pass
            
            if count == aug_multiple:
                break
        return r_coords
    
    def augment(self, multiple):
        csv_data = self.csv_data.loc[:, ['category_id', 'Filename',	'X0', 'Y0', 'X1', 'Y1',	'Border', 'Field']]
        csv_data.drop(csv_data.loc[csv_data['Field']==0].index, inplace=True)    # darkness field drop
        csv_data.drop(['Field'], axis=1, inplace=True)
        origin_train = pd.DataFrame(columns=csv_data.columns)

        train_origin_dir = self.train_dir
        train_aug_dir = self.dest_dir + f'train_aug{multiple}_cls/'
        os.makedirs(train_aug_dir, exist_ok=True)

        c, f, x0, y0, x1, y1, b = [], [], [], [], [], [], []

        aug_list = os.listdir(train_aug_dir)
        img_list = os.listdir(train_origin_dir)
        img_list.sort()

        axes = [0]    # vflip: 0, hflip: 1, v+h_flip: -1
        r_coords = self.crop_coords(multiple)
        for img in tqdm(img_list):
            file_path = train_origin_dir + img
            origin_img = cv2.imread(file_path)
            rand_y = random.randint(1, 100)
            if img not in aug_list:
                shutil.copy(file_path, train_aug_dir + img)
            inst = csv_data[csv_data['Filename']==img].reset_index(drop=True)
            origin_train = origin_train.append(inst, ignore_index=True)
            c_id_list = list(inst['category_id'])
            if sum([c_id_list[i] in [4,8,13,14,19,21] for i in range(len(c_id_list))]) == 0:
                continue

            # flip
            for axis in axes:
                flip_img = cv2.flip(origin_img, axis)
                img_id = img.split('.')[0]
                new_img_id = img_id + f'_f{axis}.jpg'
                if new_img_id not in aug_list:
                    cv2.imwrite(train_aug_dir + new_img_id, flip_img)
                for i in range(len(inst)):
                    c_i, f_i, x0_i, y0_i, x1_i, y1_i, b_i = inst.iloc[i,:]
                    c.append(c_i)  # cateory_id
                    f.append(new_img_id)
                    new_X0, new_Y0, new_X1, new_Y1 = self.flip_GT(origin_img, axis, x0_i, y0_i, x1_i, y1_i)
                    x0.append(new_X0)
                    y0.append(new_Y0)
                    x1.append(new_X1)
                    y1.append(new_Y1)
                    b.append(b_i)
            # crop
            if 13 not in c_id_list and 14 not in c_id_list and 19 not in c_id_list and 21 not in c_id_list:
                coords_len = int(len(r_coords) / 2)
            elif 13 not in c_id_list and 14 not in c_id_list:
                coords_len = int(len(r_coords) / 1)
            else:
                coords_len = len(r_coords)    
            h, w = origin_img.shape[:2]
            for r in range(coords_len):
                rx = r_coords[r][0]
                ry = r_coords[r][1]
                cropped_img = origin_img[ry:h, rx:w]  if inst.loc[0,'Border'] == 0 else origin_img[ry:h, :]
                resized_cropped_img = cv2.resize(cropped_img, (w, h), interpolation = cv2.INTER_LINEAR)
                img_id = img.split('.')[0]
                new_img_id = img_id + f'_c{r}.jpg'
                if new_img_id not in aug_list:
                    cv2.imwrite(train_aug_dir + new_img_id, resized_cropped_img)
                for i in range(len(inst)):
                    c_i, f_i, x0_i, y0_i, x1_i, y1_i, b_i = inst.iloc[i,:]
                    new_X0 = max(0, x0_i - rx) if inst.loc[0,'Border'] ==0 else x0_i
                    new_Y0 = max(0, y0_i - ry)
                    new_X1 = x1_i - rx if inst.loc[0,'Border'] == 0 else x1_i
                    new_Y1 = y1_i - ry
                    if new_X1 < 0 or new_Y1 < 0:
                        continue
                    new_x0, new_y0, new_x1, new_y1 = self.rescale_GT(cropped_img, new_X0, new_Y0, new_X1, new_Y1, 2048, 1000)
                    c.append(c_i)  # cateory_id
                    f.append(new_img_id)
                    x0.append(new_X0)
                    y0.append(new_Y0)
                    x1.append(new_X1)
                    y1.append(new_Y1)
                    b.append(b_i)
            print(f'-----{img} is augmented-----')

        new_train = pd.DataFrame(
            {'category_id':c,
            'Filename':f,
            'X0':x0,
            'Y0':y0,
            'X1':x1,
            'Y1':y1,
            'Border':b}
        )

        total_train = pd.concat([origin_train, new_train], ignore_index=True)
        total_train.to_csv(f'./MSoS_crop_vflip{multiple}.csv')

        # annotation json file
        aug_img_list = list(set(total_train['Filename'].values))
        os.makedirs(self.annotation_dir, exist_ok=True)
        self.annotation('train_aug', aug_img_list, train_aug_dir, self.annotation_dir)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--multiple", dest="multiple", default=10)
    (options, args) = parser.parse_args()
    multiple = int(options.multiple) 

    dataset=Dataset()
    dataset.split_annot()
    dataset.augment(multiple)