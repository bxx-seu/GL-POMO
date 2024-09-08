##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = True
CUDA_DEVICE_NUM = 0
USE_WANDB = False
##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "../../..")  # for problem_def
sys.path.insert(0, "../../../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSPTester import TSPTester as Tester

Projection_path = './'

run_type = 'test' # test, val
##########################################################################################
model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',

    # New graph embedding
    'sub_graph_emb': True,
    'sub_graph_steps': 1,

    # L-Encoder settings
    'local_feature': True,
    'embedding_dim_local': 32,
    'qkv_dim_local': 8,
    'head_num_local': 4,
}

### Test datasets
test_distributions = ['uniform', 'cluster', 'mixed', 'expansion', 'explosion', 'rotation']
Ns = [100,150,200,300]
test_datasets = []
test_size_list = []
test_datasets_labels = []
for d in test_distributions:
    test_datasets.extend([[Projection_path+f'data/test/unit-tsp{N}_{d}_test.pkl'] for N in Ns])
    test_size_list.extend([[N] for N in Ns])
    test_datasets_labels.extend([f'tsp{N}_{d}_test' for N in Ns])
test_pomo_size = Ns * len(test_distributions)
test_episodes_list = [1000]*len(test_datasets)

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'env_params': {
        'data_labels': test_datasets_labels,
        'data_type': 'others',
        'size_list': test_size_list,
        'data_path': test_datasets,
        'pomo_size': test_pomo_size,
        'aug_strategy': 'symnco',
        'K': 20,
    },
    'test_episodes': test_episodes_list,
    'aug_factor': 8,
    'test_batch_size': [500,336,256,128]*len(test_distributions),
    'model_type': 'ours0', # sym-nco pomo ours ours0
    'model_load': {
        'path': ['./pretrained_model/Our-UCM-CSize-cp2010.pt'],  # directory path of pre-trained model and log files saved.
    }
}




logger_params = {
    'log_file': {
        'desc': f'train__tsp_n100',
        'filename': 'run_log'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=tester_params['env_params'],
                     model_params=model_params,
                     tester_params=tester_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _set_debug_mode():
    global trainer_params
    global tester_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4
    tester_params['test_episodes'] = [10]
    tester_params['aug_factor'] = 2
    tester_params['test_batch_size'] = [4]
    tester_params['aug_batch_size'] = 4

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
