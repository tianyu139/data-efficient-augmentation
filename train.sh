export CUDA_VISIBLE_DEVICES=0

python main.py \
test_results \
-R 5 \
-s 0.3 \
-b 64 \
--dataset caltech256 \
--lr 1e-3 \
--epochs 40 \
--pretrained \
--subset_algo coreset \
--subset_main_dataset \
--enable_subset_augment \
