cd buffer
python3 buffer.py \
--dataset=ImageNet \
--model=ConvNetD5 \
--train_epochs=100 \
--num_experts=100 \
--res=128 \
--subset=imagenette \
--buffer_path="../buffer_storage_1/in1k" \
--data_path="../dataset/imagenet" \
--rho_max=0.01 \
--rho_min=0.01 \
--alpha=0.3 \
--lr_teacher=0.01 \
--mom=0. \
--batch_train=256