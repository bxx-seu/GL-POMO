
import torch
# torch.backends.cudnn.enabled = False

import os
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from TSPModel_symnco import TSPModel as Model_SymNCO
from TSPModel_ours0 import TSPModel as Model_ours0
from utils.utils import *
import copy
import pickle


class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()

        # torch.set_num_threads(5)
        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        # self.env = Env(**self.env_params)
        if self.tester_params['model_type'] == 'pomo':
            self.model = Model(**self.model_params)
        elif self.tester_params['model_type'] == 'sym-nco':
            self.model = Model_SymNCO(**self.model_params)
        elif self.tester_params['model_type'] == 'ours0':
            self.model = Model_ours0(**self.model_params)
        else:
            assert False, f"Not that model type: {self.trainer_params['model_type']}"

        # utility
        self.time_estimator = TimeEstimator()
        self.greedy_res = []
        self.sampling_res = []
        self.best_tour = []
        self.best_tour_cost = []

    @torch.no_grad()
    def run(self):

        res = {}
        self.logger.info('Test result_folder: {}'.format(self.result_folder+'.res'))

        for mp in self.tester_params['model_load']['path']:
            # Restore
            if not mp.endswith('.pt'):
                model_load = {
                    'problem_size': self.tester_params['model_load']['problem_size'],
                    'path': mp,
                    'epoch': self.tester_params['model_load']['epoch'],
                }
                checkpoint_fullname = f'{model_load["path"]}/n{model_load["problem_size"]}-checkpoint-{model_load["epoch"]}.pt' \
                    if not model_load["path"].endswith('.pt') else model_load["path"]
            else:
                model_load = {'path': mp}
                checkpoint_fullname = mp
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Loaded {}".format(checkpoint_fullname))

            start_time = time.time()

            val_score = validate(self.model, self.tester_params, model_load)

            for k in val_score.keys():
                if k not in res:
                    res[k] = []
                res[k].append(val_score[k])
            # for label in self.tester_params['env_params']['data_labels']:
            #     res[label].append(val_score[label])

            self.logger.info("Tested {}, Time: {:.2f}m".format(checkpoint_fullname, (time.time()-start_time)/60.))

        pickle.dump(res, open(self.result_folder+'.res', 'wb'))


    def _test_one_batch(self, batch_size):

        # Augmentation
        ###############################################
        aug_factor = self.tester_params['aug_factor']

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)

  

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.sub_tours_size)
        # shape: (augmentation, batch, pomo)

        no_pomo_score = -aug_reward[0, :, 0].mean()

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        self.greedy_res.extend((-aug_reward[0, :, 0]).cpu().numpy().tolist())
        self.sampling_res.extend((-max_aug_pomo_reward).float().cpu().numpy().tolist())

        # Restore the best tour
        reward = reward.view(aug_factor, batch_size, self.env.sub_tours_size)\
            .permute(1, 2, 0).reshape(batch_size,-1)
        max_score,max_idx = reward.max(dim=-1)
        # print(self.env.selected_node_list.shape)
        tour_list = self.env.selected_node_list.reshape(aug_factor, batch_size, self.env.sub_tours_size, -1).permute(1, 2, 0, 3)\
            .reshape(batch_size, aug_factor * self.env.sub_tours_size, -1) # Batch*aug_factor,POMO,N
        tour_idx = max_idx[:,None,None].expand(batch_size,1,tour_list.shape[-1])
        best_tour = tour_list.gather(dim=1, index=tour_idx).squeeze()
        self.best_tour.extend(best_tour.cpu().numpy().tolist())
        max_score = - max_score
        self.best_tour_cost.extend(max_score.cpu().numpy().tolist())

        return no_pomo_score.item(), no_aug_score.item(), aug_score.item()

def validate(model, val_params, model_load):
    val_params_ = copy.deepcopy(val_params)
    val_res = {}

    for dt_i,data_path_l in enumerate(val_params['env_params']['data_path']):
        val_params_['env_params']['data_path'] = data_path_l
        val_params_['env_params']['size_list'] = val_params['env_params']['size_list'][dt_i]
        val_params_['env_params']['data_labels'] = val_params['env_params']['data_labels'][dt_i]
        val_params_['test_episodes'] = val_params['test_episodes'][dt_i]
        val_params_['test_batch_size'] = val_params['test_batch_size'][dt_i]
        val_params_['env_params']['pomo_size'] = val_params['env_params']['pomo_size'][dt_i]

        env_test = Env(**val_params_['env_params'])

        # cuda
        USE_CUDA = val_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = val_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Augmentation
        ###############################################
        episodes = val_params_['test_episodes']
        aug_factor = val_params_['aug_factor']
        batch_size = val_params_['test_batch_size']
        # pomo_size = val_params_['env_params']['pomo_size']

        best_tour_cost = []
        best_tour = []

        # Ready
        ###############################################
        start_time = time.time()
        model.eval()
        env_test.load_raw_data()
        assert len(env_test.problem_size_list.keys()) == len(val_params_['env_params']['size_list'])
        pomo_size = env_test.problem_size_list[val_params_['env_params']['size_list'][0]]
        max_aug_pomo_rewards = []
        with torch.no_grad():
            test_num_episode = val_params_['test_episodes']
            episode = 0
            epoch_offset = 0
            while episode < test_num_episode:
                remaining = test_num_episode - episode
                batch_size = min(val_params_['test_batch_size'], remaining)

                env_test.load_problems(batch_size=batch_size,aug_factor=val_params_['aug_factor'], no_shuffle=True)
                episode += batch_size

                reset_state, reward, done = env_test.reset(test=True,pomo_size=pomo_size)
                model.pre_forward(reset_state)

                # POMO Rollout
                ###############################################
                state, reward, done = env_test.pre_step()
                while not done:
                    # predict the next node
                    selected, _ = model(state)
                    # shape: (batch, pomo)
                    state, reward, done = env_test.step(selected)

                # Return
                ###############################################
                aug_reward = reward.reshape(aug_factor, batch_size, pomo_size)
                # shape: (augmentation, batch, pomo)
                max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
                # shape: (augmentation, batch)
                max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
                # shape: (batch,)
                max_aug_pomo_rewards.append(max_aug_pomo_reward)

                # Restore best tour and cost
                reward = reward.view(aug_factor, batch_size, pomo_size) \
                    .permute(1, 2, 0).reshape(batch_size, -1)
                max_score, max_idx = reward.max(dim=-1)
                tour_list = env_test.selected_node_list.reshape(aug_factor, batch_size, pomo_size, -1).permute(
                    1, 2, 0, 3) \
                    .reshape(batch_size, aug_factor * pomo_size, -1)  # Batch*aug_factor,POMO,N
                tour_idx = max_idx[:, None, None].expand(batch_size, 1, tour_list.shape[-1])
                _best_tour = tour_list.gather(dim=1, index=tour_idx).squeeze(dim=1)
                best_tour.extend(_best_tour.cpu().numpy().tolist())
                max_score = - max_score
                best_tour_cost.extend(max_score.cpu().numpy().tolist())

        running_time = time.time() - start_time

        # Restore best tour and cost
        results = list(zip(best_tour_cost, best_tour,
                           [running_time / test_num_episode for _ in range(len(best_tour))]))
        device = 'gpu' if val_params_['use_cuda'] else os.cpu_count()
        data_path = val_params_['env_params']['data_path'][0]
        results_dir = os.path.join(os.path.dirname(data_path), 'solution', os.path.split(model_load['path'])[-1])
        os.makedirs(results_dir, exist_ok=True)
        dataset_basename, _ = os.path.splitext(os.path.split(data_path)[-1])
        solver = 'Our'
        out_file = os.path.join(results_dir, "{}-{}{}".format(
                dataset_basename,
                solver, '.pkl'))
        pickle.dump((results,device), open(out_file, 'wb'), pickle.HIGHEST_PROTOCOL)

        aug_score = -torch.cat(max_aug_pomo_rewards, dim=0).float().mean()  # negative sign to make positive value
        aug_gap = None
        data_path_size_idx = val_params_['env_params']['size_list'][0]
        if env_test.raw_data_tour_lens[data_path_size_idx] is not None:
            aug_gap = (-torch.cat(max_aug_pomo_rewards, dim=0).float() - env_test.raw_data_tour_lens[data_path_size_idx]) /  env_test.raw_data_tour_lens[data_path_size_idx] * 100

        val_res[val_params_['env_params']['data_labels']] = aug_score.item()
        if aug_gap is not None:
            val_res[val_params_['env_params']['data_labels'] + '_gap'] = aug_gap.mean().item()
            print(val_params_['env_params']['data_labels'] + '（Optimal Gap）: ' + '%.4f' % aug_gap.mean().item() + ' %', 'Running Time: ' + '%.4f' % running_time + ' s')
        del env_test
    return val_res