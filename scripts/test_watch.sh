
# # Their pre-trained model of goal inf
# python watch/predicate-train.py \
# --gpu_id 1 \
# --batch_size 4 \
# --demo_hidden 512 \
# --model_type lstmavg \
# --dropout 0 \
# --inputtype graphinput \
# --inference 1 \
# --single 0 \
# --resume 'checkpoints/demo2predicate-best_model.ckpt' \
# --checkpoint checkpoints/test




# Our training of goal inf
python watch/predicate-train.py \
--gpu_id 1 \
--batch_size 4 \
--demo_hidden 512 \
--model_type lstmavg \
--dropout 0 \
--inputtype graphinput \
--inference 1 \
--single 0 \
--use_data_parallel 0 \
--resume 'checkpoints/test/train_try/demo2predicate-best_model.ckpt' \
--checkpoint checkpoints/test/train_try


