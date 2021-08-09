#!/bin/bash

atc --singleop=config/add_op.json --soc_version=Ascend910 --output=op_models

python acl_execute_add.py

