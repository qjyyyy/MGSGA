# MGSGA
The implementation of paper **MGSGA: Multi-grained and Semantic-guided Alignment for Text-Video Retrieval**.
![alt](URL "title")

## Updates
[2023/11/07]: We will release the code asap. (I am busy with other DDLs. After that, I will open the source code as soon as possible. Please understand.)


## Quick Start
### Train on MSRVTT
```python  
DATA_PATH=YOUR DATA_PATH
python -m torch.distributed.launch --nproc_per_node=2  --master_port 3256687 \
main_task_retrieval.py --do_train --num_thread_reader=0 \
--epochs=5 --batch_size=32 --n_display=50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--features_path ${DATA_PATH}/MSRVTT_Videos \
--output_dir ckpts/ckpt_msrvtt_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 128 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/16 \
--loss_type cross_enc \
--interaction hybrid
```
## Acknowledgements
This code implementation are adopted from [CLIP](https://github.com/openai/CLIP "CLIP") and [DRL](https://github.com/foolwood/DRL "DRL"). We sincerely appreciate for their contributions.
