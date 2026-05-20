uv run python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --datasets tartan_drive \
    --batch_size 32 \
    --num_workers 8 \
    --eval_type rollout \
    --output_dir ./results \
    --gt 1