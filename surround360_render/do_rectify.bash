#!/bin/bash

./bin/TestRingRectification \
--rig_json_file ./res/config/17cmosis_default.json \
--src_intrinsic_param_file ./res/config/sunex_intrinsic.xml \
--output_transforms_file ./rectify.yml \
--root_dir ~/Downloads/sample_dataset \
--frames_list 000000 \
--visualization_dir ./rectify_vis

