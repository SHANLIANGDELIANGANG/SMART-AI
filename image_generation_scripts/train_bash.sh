export CUDA_VISIBLE_DEVICES=1,2,5,6
export OPENAI_LOGDIR=./logs/exp_1

NUM_GPUS=4

mpiexec -n $NUM_GPUS python image_generation_scripts/image_train.py \
    