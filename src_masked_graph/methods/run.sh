
if [[ "$1"  == "swinir" ]]; then

    CUDA_VISIBLE_DEVICES=1,3 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 \
   swinir_ppd.py --opt ../deep_learning/utils/swinir_config.json  --dist True > swinir_ppd.txt

fi




