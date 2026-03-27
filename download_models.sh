#!/bin/bash

REMOTE_USER=root
REMOTE_HOST=213.173.107.8
PORT=14551
KEY=~/.ssh/id_ed25519

REMOTE_PATH=/workspace
LOCAL_PATH=/Users/emilfahretdinov/msc_hse/models_trained/fno_delta_lr1e-3_20ep_nmodes64_nlayers4

# create local directory if not exists
mkdir -p "$LOCAL_PATH"

echo "Downloading files..."

scp -P $PORT -i $KEY $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/best_by_valid_1step_vrmse_delta.pt $LOCAL_PATH/
scp -P $PORT -i $KEY $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/best_by_valid_rollout_vrmse_delta.pt $LOCAL_PATH/
scp -P $PORT -i $KEY $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/final_model_delta.pt $LOCAL_PATH/

echo "Done!"