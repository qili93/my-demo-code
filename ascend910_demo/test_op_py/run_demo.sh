#!/bin/bash

# step 1: use atc tool to generate om model
atc --singleop=config/test_op.json --soc_version=Ascend910 --output=op_models

# step 2: execute op by acl_execute_op.py
python acl_execute_op.py
