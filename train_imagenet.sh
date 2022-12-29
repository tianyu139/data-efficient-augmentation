export CUDA_VISIBLE_DEVICES=0

python main.py \
test_results \
-R 15 \
-s 0.5 \
-b 64 \
--dataset imagenet \
--arch resnet50 \
--lr 1e-1 \
--epochs 90 \
--subset_algo coreset \
--subset_main_dataset \
--enable_subset_augment \
--override_subset_main_random
