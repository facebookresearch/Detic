from subprocess import Popen
import sys
import os 

procs = []
for i in range(6):
    script = "demo_box.py"
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = str(i)
    procs.append(Popen(['python', script, '--config-file', 'configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml','--input','/home/zeyu/brute_force/agg_{}/*/*.jpg'.format(i), '--output','/home/zeyu/brute_force/detection_result_images/download_data{}/'.format(i), '--vocabulary','custom','--custom_vocabulary','box,envelope,package,person,monitor,car,tree,','--opts','MODEL.WEIGHTS','models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth' ],env = my_env))

for proc in procs:
    proc.wait()
    
