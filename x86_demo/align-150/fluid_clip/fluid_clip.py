import os
import time
import argparse
import functools
import numpy as np
import paddle.fluid as fluid
from paddle.fluid import core

from utility import add_arguments
from utility import print_arguments

np.random.seed(10)
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('dst_model_dir', str, '../assets/models/align150-fp32-dst/', 'The modified fluid model dir.')
add_arg('dst_model_filename', str, '__model__', 'The modified fluid model file name.')
add_arg('dst_params_filename', str, '__params__', 'The modified fluid params file name.')
add_arg('src_model_dir', str, '../assets/models/align150-fp32/', 'The fluid model dir.')
add_arg('src_model_filename', str,  '__model__', 'The fluid model file name.')
add_arg('src_params_filename', str,  '__params__', 'The fluid params file name.')

#parser
ARGS = parser.parse_args()
print_arguments(ARGS)

def main(argv=None):
    # load fluid model
    print('Loading fluid model...')
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    [test_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(ARGS.src_model_dir, exe, model_filename=ARGS.src_model_filename, params_filename=ARGS.src_params_filename)
    #[test_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(ARGS.src_model_dir, exe)
    print('--- feed_target_names ---')
    print(feed_target_names)
    print('--- fetch_targets ---')
    print(fetch_targets)
    try:
        os.makedirs(ARGS.dst_model_dir)
    except OSError as e:
        if e.errno != 17:
            raise
    feed_target_names = ["VISface_landmark_0"]
    fetch_targets = [test_program.current_block().var("VISface_landmark_22968.avg_pool.output.1.tmp_0")]
    fluid.io.save_inference_model(ARGS.dst_model_dir, feed_target_names, fetch_targets, exe, test_program, ARGS.dst_model_filename, ARGS.dst_params_filename)
    #fluid.io.save_inference_model(ARGS.dst_model_dir, feed_target_names, fetch_targets, exe, test_program)

if __name__ == '__main__':
    main()
