#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=/home/vedatb/batch_outputs/%j.out
#SBATCH --error=/home/vedatb/batch_outputs/%j.error

output_path="/home/$USER/datasets/v1/masks.hdf5"
img_dir="/home/$USER/GQA/images"
scene_graph="/home/$USER/GQA/scene_graph.json"

echo "detectron2"
srun python generate_masks.py detectron2 --img-dir $img_dir --scene-graph $scene_graph  --output $output_path --use-gpu
sleep 60

echo "detic"
srun python generate_masks.py detic --img-dir $img_dir --scene-graph $scene_graph --output $output_path --use-gpu
sleep 60

echo "detic + custom vocabulary + attributes"
srun python generate_masks.py detic --img-dir $img_dir --scene-graph $scene_graph --output $output_path --custom-vocabulary --include-attributes --use-gpu
