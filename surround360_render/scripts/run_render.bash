#!/bin/bash

python2 run_all.py --steps_render --data_dir ~/Downloads/sample_dataset --dest_dir ~/Downloads/sample_dataset --rectify_file $PWD/rectify.yml --verbose --enable_top --enable_bottom --save_debug_images
