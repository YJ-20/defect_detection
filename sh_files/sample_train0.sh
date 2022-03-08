# <<caution>> run sh files at root directory of this repository.
# run : nohup sh ./sh_files/sample_train0.sh &
# kill : ps -ef | grep sample_train0.sh
#		 kill -9 ######
for mode in 'train'

do

for trial_no in 0

do

for max_iter in 500000

do

for gpu_no in 6

do

for weight_decay in 0.001

do

for multiple in 10

do

for ims_per_batch in 12

do

for backbone in 'ResNext-101'

do

	CUDA_VISIBLE_DEVICES=${gpu_no} python3 ./main.py --mode ${mode} --trial_no ${trial_no} --max_iter ${max_iter} --gpu_no ${gpu_no} --weight_decay ${weight_decay} --multiple ${multiple} --ims_per_batch ${ims_per_batch} --backbone ${backbone} > ./log/${mode}${trial_no}_gpu${gpu_no}_b${backbone}_wd${weight_decay}_i${max_iter}.log 2>&1

done

done

done

done

done

done

done

done