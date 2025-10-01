python test_net.py \
    --config-file configs/brats21_supervision/faster_rnn_R_50_FPN_sup5_run1.yaml \
    --model-weights /path/to/best/checkpoint \
    --test-dataset-name "brats_test_dataset" \
    --test-annotations /path/to/instance_test.json \
    --test-images /path/to/test/iamges \
    --thing-classes tumor \
    --output-dir /path/to/output/test_results
