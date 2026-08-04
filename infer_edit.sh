cd /datasets/codes_zsqiao/SenseNova-U1

# python examples/editing/inference.py \
#     --model_path /models/SenseNova-U1.5-8B-MoT-Preview-Layered \
#     --image /datasets/codes_zsqiao/SenseNova-U1/examples/editing/0978_将红色方框中的文字修改为：_拉屎去吧_，然后去掉红色方框_col2.jpg \
#     --prompt "将红色方框中的文字修改为：“拉屎去吧”，然后去掉红色方框" \
#     --no-use-edit-pe \
#     --input_max_pixels 4194304 \
#     --target_pixels 16777216 \
#     --cfg_scale 4.0 \
#     --img_cfg_scale 1.0 \
#     --cfg_norm none \
#     --timestep_shift 3.0 \
#     --num_steps 50 \
#     --attn_backend auto \
#     --device cuda \
#     --dtype bfloat16 \
#     --output outputs/edit.png \
#     --profile



# python examples/editing/inference.py \
#     --model_path /models/SenseNova-U1.5-8B-MoT-Preview-Layered \
#     --image /datasets/codes_zsqiao/FlashNFT/dataset/dpo_v2/00319_将红色方框中的文字修改为：_长兴明德电子科技_，然后去掉红色方框_col2_1_original.jpeg \
#     --prompt "将红色方框中的文字修改为：“长兴明德电子科技”，然后去掉红色方框" \
#     --no-use-edit-pe \
#     --input_max_pixels 4194304 \
#     --target_pixels 4194304 \
#     --cfg_scale 4.0 \
#     --img_cfg_scale 1.0 \
#     --cfg_norm none \
#     --timestep_shift 3.0 \
#     --num_steps 50 \
#     --attn_backend auto \
#     --device cuda \
#     --dtype bfloat16 \
#     --output outputs/edit.png \
#     --profile


python examples/editing/inference.py \
    --model_path /models/SenseNova-U1.5-8B-MoT-Preview-Layered-step9000 \
    --image /datasets/codes_zsqiao/SenseNova-U1/outputs/11111488/full_image_with_bboxes.png \
    --prompt "把图中红框选中的所有元素分层，并移除红框。" \
    --num_layers 3 \
    --no-use-edit-pe \
    --input_max_pixels 4194304 \
    --target_pixels 4194304 \
    --cfg_scale 4.0 \
    --img_cfg_scale 1.0 \
    --cfg_norm none \
    --timestep_shift 3.0 \
    --num_steps 50 \
    --attn_backend auto \
    --device cuda \
    --dtype bfloat16 \
    --output outputs/pred.png \
    --profile