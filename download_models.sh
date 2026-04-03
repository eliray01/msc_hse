#!/bin/bash

REMOTE_USER=root
REMOTE_HOST=213.173.110.111
PORT=17806
KEY=~/.ssh/id_ed25519

REMOTE_PATH=/workspace
LOCAL_PATH=/Users/emilfahretdinov/msc_hse/models_trained/unet_delta_4_2_1_42_7

# create local directory if not exists
mkdir -p "$LOCAL_PATH"

echo "Downloading files..."

scp -P $PORT -i $KEY $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/best_cnextunet_delta_by_valid_1step_vrmse.pt $LOCAL_PATH/
scp -P $PORT -i $KEY $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/best_cnextunet_delta_by_valid_rollout_vrmse.pt $LOCAL_PATH/
scp -P $PORT -i $KEY $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/final_cnextunet_full_frame.pt $LOCAL_PATH/

echo "Done!"