##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = True
CUDA_DEVICE_NUM = 0
USE_WANDB = False and not DEBUG_MODE
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

from TSPTrainer_ours import TSPTrainer as Trainer

Projection_path = './'
##########################################################################################
# parameters
env_params = {
    'data_type': 'distribution', # 'size-distribution',
    'size_list': [69,83,100,120,144],
    'data_path': [Projection_path+'data/train/'+datafile for datafile in ['unit-tsp69_UCMixed_train_mix_20w.pkl',
                                                                          'unit-tsp83_UCMixed_train_mix_20w.pkl',
                                                                          'unit-tsp100_UCMixed_train_mix_20w.pkl',
                                                                          'unit-tsp120_UCMixed_train_mix_20w.pkl',
                                                                          'unit-tsp144_UCMixed_train_mix_20w.pkl']],
    'problem_size': 100,
    'pomo_size': 50 * 2,
    'aug_strategy': 'symnco',
    'K': 20,
}

logger_params = {
    'log_file': {
        'desc': f'train__tsp_n{env_params["problem_size"]}',
        'filename': 'run_log',
    }
}

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

    # L-Encoder setting
    'local_feature': True,
    'embedding_dim_local': 32,
    'qkv_dim_local': 8,
    'head_num_local': 4,
}

#####################################################
optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [2001],
        'gamma': 0.1
    }
}
#####################################################

trainer_params = {
    'wandb_name':'Sym-NCO+SSSL+Local(K20)+UCMixed-CSize',
    'epochs': 2010,
    'sr_size': 2,
    'model_type': 'ours0', # 'pomo', 'sym-nco', 'ours0'
    'steps_num': 3,
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': './result/20240708_160857_train__tsp_n100',  # directory path of pre-trained model and log files saved.
        'problem_size': 120,
        'epoch': 1820,  # epoch version of pre-trained model to laod.
    },

    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,

    'train_episodes': 100 * 1000,
    'train_batch_size': 64,
    'alpha':0.1,
    'validate': False,

    'wandb': USE_WANDB,
    'seed': 2024,
    'logging': {
        'model_save_interval': 5,
        'img_save_interval': 5,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_20.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
}


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'env_params': {
        'data_labels': ['tsp100_cluster_val',],
        'data_path': [Projection_path+'data/val/unit-tsp100_cluster_val.pkl',],
    },
    'test_episodes': [1000],
    'aug_factor': 8,
    'test_batch_size': [1000],
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params,tester_params=tester_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    global tester_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4
    tester_params['test_episodes'] = 10
    tester_params['aug_factor'] = 2
    tester_params['test_batch_size'] = 4
    tester_params['aug_batch_size'] = 4

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
