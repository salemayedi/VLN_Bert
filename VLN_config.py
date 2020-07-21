from types import SimpleNamespace
config = SimpleNamespace(
    from_pretrained="save/multitask_model/multi_task_model.bin",
    bert_model="bert-base-uncased",
    config_file="config/bert_base_6layer_6conect.json",
    # max_seq_length=101,
    train_batch_size=1,
    do_lower_case=True,
    predict_feature=False,
    seed=42,
    num_workers=0,
    baseline=False,
    img_weight=1,
    distributed=False,
    objective=1,
    visual_target=0,
    dynamic_attention=False,
    task_specific_tokens=True,
    tasks='1',
    save_name='',
    in_memory=False,
    local_rank=-1,
    split='mteval',
    clean_train_sets=True,
    gradient_accumulation_steps=1,
    num_train_epochs=10.0,
    start_epoch=0,
    without_coattention=False,
    learning_rate=1e-4,
    adam_epsilon=1e-8,
    warmup_proportion=0.1,
    # feature extractor
    threshold_similarity=0.7,
    best_features=10,
    max_temporal_memory_buffer=3,  # the last pic is included, so you are basically only comparing to max_temporal_memory_buffer-1
    # track temporal features
    track_temporal_features=True,
    mean_layer=True,
                # if true output feature extractor embedding [m * 2048];
                # if False  output feature extractor embedding [ 2048];
    num_key_frames=4,
    num_boxes=10
)
