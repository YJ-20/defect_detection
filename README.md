# defect_detection

This repo contains the training and evaluation code for the project "Detecting steel defects"

This code is based on [Detectron2](https://github.com/facebookresearch/detectron2) but modified and refactored to realize defect detections.

## Requirements

* Prepare steel defects data. If you download them to somewhere else, you need to update the variables, "src" and "csv_data_na", in prepare_data.py.

* [Optional] Create a conda or python virtual environment.

* Install required packages using the `requirements.txt` file.

    `pip install -r requirements.txt`

## Usage

run : `nohup sh ./sh_files/sample_train0.sh &` 
kill : `ps -ef | grep sample_train0.sh` 
		   `kill -9 ######`

! Note that run sh files at root directory of this repository.    

## Examples



## ToDo:

- [X] *Initial code release.*
- [ ] Update knowledge distillation code.

## Acknowledgements
<sub>This work was funded by the project "Research on optimal methodology for high-speed image AI processing". </sub>
