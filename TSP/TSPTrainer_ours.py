import torch
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from TSPModel_symnco import TSPModel as Model_SymNCO
from TSPModel_ours0 import TSPModel as Model_ours0
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
import torch.nn.functional as F
from utils.utils import *
import random
import copy

class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params,tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.tester_params = tester_params
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # Seed
        self.seed = self.trainer_params['seed']
        seed_everything(self.seed)

        # Main Components
        if self.trainer_params['model_type'] == 'pomo':
            self.model = Model(**self.model_params)
        elif self.trainer_params['model_type'] == 'sym-nco':
            self.model = Model_SymNCO(**self.model_params)
        elif self.trainer_params['model_type'] == 'ours0':
            self.model = Model_ours0(**self.model_params)
        else:
            assert False, f"Not that model type: {self.trainer_params['model_type']}"
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = f'{model_load["path"]}/n{model_load["problem_size"]}-checkpoint-{model_load["epoch"]}.pt'
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            # seed state
            torch.set_rng_state(checkpoint['torch_rng_state'].type(torch.ByteTensor))
            torch.cuda.set_rng_state(checkpoint['torch_cuda_rng_state'].type(torch.ByteTensor))
            np.random.set_state(checkpoint['np_random_state'])
            random.setstate(checkpoint['py_random_state'])

            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        model_save_interval = self.trainer_params['logging']['model_save_interval']
        img_save_interval = self.trainer_params['logging']['img_save_interval']
        self.logger.info("Start loading train dataset...")
        self.env.load_raw_data()
        self.logger.info("Train dataset is loaded.")
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            self.env.shuffle_data()
            # Train
            train_loss = self._train_one_epoch(epoch)
            # LR Decay
            self.scheduler.step()

            if self.trainer_params['validate']:
                val_score = validate(self.model, self.tester_params)
                for name in val_score.keys():
                    self.result_log.append(name, epoch, val_score[name])


            ############################
            # Logs & Checkpoint
            ############################
            self.result_log.append('train_loss', epoch, train_loss)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data(),
                    # random states
                    'py_random_state': random.getstate(),
                    'np_random_state': np.random.get_state(),
                    'torch_rng_state': torch.get_rng_state(),
                    'torch_cuda_rng_state': torch.cuda.get_rng_state(),
                }
                torch.save(checkpoint_dict, '{}/n{}-checkpoint-{}.pt'.format(self.result_folder,self.env.problem_size, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)


    def _train_one_epoch(self, epoch):

        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        self.step_num = epoch * (train_num_episode//self.trainer_params['train_batch_size'] + 1) + 1
        epoch_offset = (epoch-1)*self.trainer_params['train_episodes']  # Epoch start from 1
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            self.env.load_problems(batch_size=batch_size,aug_factor=self.trainer_params['sr_size'],
                                   raw_data_idx=self.env_params['size_list'][torch.multinomial(torch.ones(size=(len(self.env_params['size_list']),)),1)])
            avg_loss = self._train_one_batch(batch_size)
            # score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size
            self.step_num += 1
            loop_cnt += 1
            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                            loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 loss_AM.avg))

        return loss_AM.avg

    def _train_one_batch(self, batch_size):
        # Prep
        ###############################################
        self.model.train()
        reset_state,_,_ = self.env.reset()
        proj_nodes = self.model.pre_forward(reset_state,return_h_mean=True)
        # predict the next node
        ###############################################
        prob_list = torch.zeros(size=(self.env.batch_size, self.env.pomo_size, 0))
        opt_edge_dis_list = torch.zeros(size=(self.env.batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~steps_num)
        steps_num_list_idx = torch.ones(size=(self.env.problem_size-2,)).multinomial(self.trainer_params['steps_num'],replacement=False)
        steps_num_list = torch.arange(1, self.env.problem_size - 1, dtype=torch.long)[steps_num_list_idx]
        for step_n in steps_num_list:
            env_state = self.env.set_env_state(step_n)
            prob = self.model(state=env_state)
            prob_list = torch.cat((prob_list, prob.log().mean(dim=-1)[:, :, None]), dim=2)
            opt_edge_dis_list = torch.cat((opt_edge_dis_list, env_state.next_edge_len[:,:,None]), dim=2)
        loss_ssl_mean = -prob_list.sum(dim=2).mean()
        # symnco
        if self.trainer_params['sr_size']>1:

            # State Invariant
            ###############################################
            proj_nodes = proj_nodes.reshape(self.trainer_params['sr_size'], batch_size, -1)

            proj_nodes = F.normalize(proj_nodes, dim=-1)

            proj_1 = proj_nodes[0]
            proj_2 = proj_nodes[1]

            similarity_matrix = torch.matmul(proj_1, proj_2.T)
            mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool)
            positive = similarity_matrix[mask].view(similarity_matrix.shape[0],-1)
            negative = similarity_matrix[~mask].view(similarity_matrix.shape[0],-1)


            negative = torch.exp(negative).sum(dim=-1,keepdim=True)

            sim_loss = -(positive - torch.log(negative)).mean()

            loss_mean = loss_ssl_mean + self.trainer_params['alpha'] * sim_loss
        else:
            loss_mean = loss_ssl_mean

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return loss_mean.item()


def validate(model, val_params):
    val_params_ = copy.deepcopy(val_params)
    val_res = {}

    for dt_i, data_path in enumerate(val_params['env_params']['data_path']):
        val_params_['env_params']['data_path'] = data_path
        val_params_['env_params']['data_labels'] = val_params['env_params']['data_labels'][dt_i]
        val_params_['test_episodes'] = val_params['test_episodes'][dt_i]
        val_params_['test_batch_size'] = val_params['test_batch_size'][dt_i]

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

        # Ready
        ###############################################
        model.eval()
        env_test.load_raw_data()
        pomo_size = env_test.problem_size
        with torch.no_grad():
            max_aug_pomo_rewards = []
            test_num_episode = val_params_['test_episodes']
            episode = 0
            epoch_offset = 0
            while episode < test_num_episode:
                remaining = test_num_episode - episode
                batch_size = min(val_params_['test_batch_size'], remaining)

                env_test.load_problems(epoch_offset=epoch_offset, episode=episode, batch_size=batch_size,
                                       aug_factor=val_params_['aug_factor'])
                episode += batch_size

                reset_state, reward, done = env_test.reset(test=True, pomo_size=pomo_size)
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
            aug_score = -torch.cat(max_aug_pomo_rewards, dim=0).float().mean()  # negative sign to make positive value
            aug_gap = None
            if env_test.raw_data_tour_lens is not None:
                aug_gap = (-torch.cat(max_aug_pomo_rewards,
                                      dim=0).float() - env_test.raw_data_tour_lens) / env_test.raw_data_tour_lens * 100

        val_res[val_params_['env_params']['data_labels']] = aug_score.item()
        if aug_gap is not None:
            val_res[val_params_['env_params']['data_labels'] + '_gap'] = aug_gap.mean().item()
        del env_test
    return val_res