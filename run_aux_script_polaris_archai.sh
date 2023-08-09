#!/bin/bash
NUM_NODES=$1

#if [[ $PMI_RANK -ge $NUMBER_GPUS ]]; then
if [[ $PMI_RANK -ge 1 ]]; then
	cd $HOME/experiments
	./cpp-store/server \
                --thallium_connection_string "ofi+verbs"\
                --num_threads 1 \
                --num_servers 1 \
                --storage_backend "map" \
                --ds_colocated 0
else
	export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
	python3 archai/nlp/nas/zero_cost_utils/pred.py --exp_name /home/mmadhya1/archai/archai/nlp/nas/saved_logs/random_TransXL_wt103
fi
