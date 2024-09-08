
import os
from dataclasses import dataclass
import torch
import pickle
import numpy as np
from TSProblemDef import get_random_problems, augment_xy_data_by_8_fold, augment_xy_data_by_N_fold, augment_xy_data_by_N_fold_ours
from utils.utils import read_tsplib,unit_normalization

@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)
    data_knn: torch.Tensor = None
    # shape: (batch, problem, K, 3)
    knn_idx: torch.Tensor = None
    # shape: (batch, problem, K)

@dataclass
class Step_State:
    first_node: torch.Tensor = None
    current_node: torch.Tensor = None
    next_opt_node: torch.Tensor = None
    next_edge_len: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)
    length_of_sub_tour: torch.Tensor = None
    sub_tours_size: torch.Tensor = None

    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    first_node_idx: torch.Tensor = None
    problem_size: torch.Tensor = None
    opt_tours: torch.Tensor = None


class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.data_path = env_params['data_path']
        self.K = env_params['K']

        # Const @Load_Problem
        ####################################
        self.problems = None
        self.data_knn = None
        self.knn_idx = None

        self.batch_size = None
        self.pomo_size = None
        self.problem_size = None
        self.BATCH_IDX = None
        self.SUB_TOURS_IDX = None
        # IDX.shape: (batch, pomo)
        self.aug_strategy = env_params['aug_strategy']
        # shape: (batch, node, node)
        self.raw_data_nodes = {}
        self.raw_data_tours = {}
        self.raw_data_number = {}
        self.problem_size_list = {}
        self.raw_data_index = {}
        self.raw_data_index_begin = {}
        self.raw_data_tour_lens = {}

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        self.FLAG__use_saved_problems = False
        # shape: (batch, pomo, 0~problem)


    def load_raw_data(self):
        self.FLAG__use_saved_problems = True
        assert len(self.env_params['size_list']) == len(self.env_params['data_path'])
        for f_idx,filename in enumerate(self.env_params['data_path']):
            if filename.endswith('pkl'):
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                # data = [torch.FloatTensor(row) for row in data]
                raw_data_nodes = data.astype(np.float32)

                # Load opt tours
                dataset_basename, ext = os.path.splitext(os.path.split(filename)[-1])
                dataset_basename1 = 'BigInt-' + '-'.join(dataset_basename.split('-')[1:])
                tour_path = os.path.join(os.path.dirname(filename), 'BigInt', 'solution', 'concorde',
                                         '{}-{}{}'.format(dataset_basename1, 'concorde', ext))
                if os.path.isfile(tour_path):
                    opt_tours = []
                    opt_tour_lens = []
                    for sol in pickle.load(open(tour_path, 'rb'))[0]:
                        opt_tours.append(sol[1])
                        opt_tour_lens.append(sol[0] / 10000000.)
                    raw_data_tours = opt_tours
                    raw_data_tour_lens = torch.tensor(opt_tour_lens, requires_grad=False)
                else:  # For Test
                    assert False, 'The solution file of {} is not exist'.format(tour_path)
                # print('self.raw_data_tours: ', self.raw_data_tours.shape)
                # print('self.raw_data_nodes: ', self.raw_data_nodes.shape)
            elif filename.endswith('txt'):
                raw_data_tour_lens = None
                raw_data_nodes = []
                raw_data_tours = []
                for line in open(filename, "r").readlines():
                    line = line.split(" ")
                    num_nodes = int(line.index('output') // 2)
                    nodes = [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]

                    raw_data_nodes.append(nodes)
                    tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]
                    raw_data_tours.append(tour_nodes)
                raw_data_nodes = np.array(raw_data_nodes).astype(np.float32)
            elif filename.endswith('tsp'):
                # For testing tsplib
                raw_data_tour_lens = None
                raw_data_tours = None
                raw_data_nodes = []

                # just one instance
                nodes, edge_weight_type = read_tsplib(filename)
                assert 'EUC_2D' in edge_weight_type, "The edge type of TSPLIB instance is not EUC_2D."
                nodes = np.array(nodes).astype(np.float32)
                nodes = unit_normalization(nodes)

                raw_data_nodes.append(nodes)
                raw_data_nodes = np.array(raw_data_nodes)
            else:
                assert False, "Cannot load data file {}".format(self.data_path)

            if self.raw_data_nodes is None:
                self.raw_data_nodes = {}
                self.raw_data_tours = {}
                self.raw_data_number = {}
                self.problem_size_list = {}
                self.raw_data_index = {}
                self.raw_data_index_begin = {}
                self.raw_data_tour_lens = {}
            data_idx = self.env_params['size_list'][f_idx]
            self.raw_data_nodes[data_idx] = torch.tensor(raw_data_nodes, requires_grad=False)
            self.raw_data_tours[data_idx] = torch.tensor(raw_data_tours, requires_grad=False) if raw_data_tours is not None else None
            self.raw_data_number[data_idx] = self.raw_data_nodes[data_idx].shape[0]
            self.problem_size_list[data_idx] = self.raw_data_nodes[data_idx].shape[1]
            self.raw_data_index[data_idx] = torch.arange(self.raw_data_number[data_idx])
            self.raw_data_index_begin[data_idx] = 0
            self.raw_data_tour_lens[data_idx] = raw_data_tour_lens if raw_data_tour_lens is not None else None
            print('load raw dataset {} done!'.format(filename))


    def shuffle_data(self, raw_data_idx=None):
        if raw_data_idx is None:
            for raw_data_idx in self.env_params['size_list']:
                self.raw_data_index[raw_data_idx] = torch.randperm(self.raw_data_number[raw_data_idx]).long()
                self.raw_data_index_begin[raw_data_idx] = 0
            print('Env: Shuffle Raw Data')
        else:
            assert raw_data_idx in self.env_params['size_list'], "shuffle_data: raw_data_idx not in self.env_params['data_size']"
            self.raw_data_index[raw_data_idx] = torch.randperm(self.raw_data_number[raw_data_idx]).long()
            self.raw_data_index_begin[raw_data_idx] = 0
            print(f'Env: Shuffle Raw Data {str(raw_data_idx)}')
        # index = torch.randperm(len(self.raw_data_nodes)).long()
        # self.raw_data_nodes = self.raw_data_nodes[index]
        # self.raw_data_tours = self.raw_data_tours[index]

    def load_problems(self, batch_size=64, aug_factor=1, raw_data_idx=None, no_shuffle=False):
        # assert self.raw_data_number >= episode, "There is not enough samples!!"
        if raw_data_idx is None:
            raw_data_idx = self.env_params['size_list'][0]
        begin_index = self.raw_data_index_begin[raw_data_idx]
        end_index = (begin_index+batch_size) % self.raw_data_number[raw_data_idx]
        self.raw_data_index_begin[raw_data_idx] = end_index
        if end_index > begin_index:
            index = self.raw_data_index[raw_data_idx][begin_index:end_index]
        else:
            index = torch.cat([self.raw_data_index[raw_data_idx][begin_index:], self.raw_data_index[raw_data_idx][:end_index]], dim=-1)

        self.problems = self.raw_data_nodes[raw_data_idx][index]  # (B,N,2)
        self.problem_size = self.problems.shape[1]
        self.opt_tours = self.raw_data_tours[raw_data_idx][index] if self.raw_data_tours[raw_data_idx] is not None else None
        self.opt_tour_lens = self.raw_data_tour_lens[raw_data_idx][index] if self.raw_data_tour_lens[raw_data_idx] is not None else None
        self.batch_size = batch_size
        self.pomo_size = min(2 * self.problem_size, self.env_params['pomo_size'])
        # problems.shape: (batch, problem, 2)
        # self.edge_val = torch.norm(self.problems[:, :, None] - self.problems[:, None], dim=-1, p=2)

        if end_index <= begin_index and not no_shuffle:
            self.shuffle_data(raw_data_idx)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
            else:
                self.batch_size = self.batch_size * aug_factor
                if self.aug_strategy == 'symnco':
                    self.problems = augment_xy_data_by_N_fold(self.problems,aug_factor)
                else:
                    self.problems = augment_xy_data_by_N_fold_ours(self.problems, aug_factor)
            self.opt_tours = self.opt_tours.repeat(aug_factor,1) if self.opt_tours is not None else None

        self.dist = (self.problems[:, :, None, :] - self.problems[:, None, :, :]).norm(p=2, dim=-1)
        if self.K > 0:
            sorted_dist, sorted_dist_idx = self.dist.topk(self.K, dim=-1, largest=False, sorted=True)
            # (batch,N,K)
            norm_fac = (sorted_dist.max(-1)[0].unsqueeze(-1) + 1e-6)  # (batch,N,1)
            self.data_knn = sorted_dist / norm_fac # (batch,N,K)

            relative_xy = (self.problems[:, :, None, :] - self.problems[:, None, :, :]).take_along_dim(sorted_dist_idx.unsqueeze(-1).expand(
                -1,-1,-1,2), dim=-2)
            # (batch,N,K,2)
            relative_x = relative_xy[:, :, :, 0]
            relative_y = relative_xy[:, :, :, 1]

            theta = torch.atan2(relative_y, relative_x)
            # (batch,N,K)
            self.data_knn = torch.cat([self.data_knn.unsqueeze(-1), norm_fac.expand(self.data_knn.shape).unsqueeze(-1), theta.unsqueeze(-1)],
                                      dim=-1)
            self.knn_idx = sorted_dist_idx
            # (batch,N,K,3)

        return self.problems

    def set_env_state(self, step_num=None):
        # In training phase
        assert self.opt_tours is not None, "Need opt_tours to set env state"
        if step_num is not None:
            length_of_sub_tour = step_num
        else:
            length_of_sub_tour = torch.randint(low=1,high=self.problem_size-1,size=(1,))[0]
        sub_tours_size = self.pomo_size
        half_sub_tours_num = self.pomo_size // 2
        assert sub_tours_size>0 and sub_tours_size % 2 == 0, "sub_tours_size % 2 ！= 0"

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, sub_tours_size)
        self.SUB_TOURS_IDX = torch.arange(sub_tours_size)[None, :].expand(self.batch_size, sub_tours_size)

        idx0 = torch.ones(size=(self.problem_size,)).multinomial(half_sub_tours_num, replacement=False)
        idx1 = torch.ones(size=(self.problem_size,)).multinomial(half_sub_tours_num, replacement=False)
        PROBLEM_IDX = torch.arange(0,self.problem_size).long()
        self.first_node_idx = torch.cat((PROBLEM_IDX[idx0], PROBLEM_IDX[idx1]),dim=-1)[None,:].expand(self.batch_size, sub_tours_size)
            # torch.arange(self.problem_size)[None, :].expand(self.batch_size, self.problem_size).repeat(1, 2)

        first_node = self.opt_tours[self.BATCH_IDX, self.first_node_idx]
        current_node1 = self.opt_tours[self.BATCH_IDX[:, :half_sub_tours_num], (self.first_node_idx[:, :half_sub_tours_num] + length_of_sub_tour - 1) % self.problem_size]
        current_node2 = self.opt_tours[self.BATCH_IDX[:, half_sub_tours_num:], self.first_node_idx[:, half_sub_tours_num:] - length_of_sub_tour + 1]
        current_node = torch.cat((current_node1, current_node2), dim=-1)
        ninf_mask = torch.zeros((self.batch_size, sub_tours_size, self.problem_size))
        ninf_mask[self.BATCH_IDX[:, :half_sub_tours_num, None], self.SUB_TOURS_IDX[:, :half_sub_tours_num, None],
                  torch.stack(
                      [self.opt_tours[self.BATCH_IDX[:, :half_sub_tours_num], (self.first_node_idx[:, :half_sub_tours_num] + i) % self.problem_size] for
                       i in range(length_of_sub_tour)],
                      dim=-1)] = float('-inf')
        ninf_mask[self.BATCH_IDX[:, half_sub_tours_num:, None], self.SUB_TOURS_IDX[:, half_sub_tours_num:, None],
                  torch.stack([self.opt_tours[self.BATCH_IDX[:, half_sub_tours_num:], self.first_node_idx[:, half_sub_tours_num:] - i] for i in
                               range(length_of_sub_tour)],
                              dim=-1)] = float('-inf')

        if length_of_sub_tour == 1:
            next_opt_node1 = self.opt_tours[self.BATCH_IDX, (self.first_node_idx + length_of_sub_tour) % self.problem_size]
            next_opt_node2 = self.opt_tours[self.BATCH_IDX, self.first_node_idx - length_of_sub_tour]
            next_opt_node = torch.stack((next_opt_node1, next_opt_node2), dim=-1)
        else:
            next_opt_node1 = self.opt_tours[self.BATCH_IDX[:, :half_sub_tours_num], (
                    self.first_node_idx[:, :half_sub_tours_num] + length_of_sub_tour) % self.problem_size]
            next_opt_node2 = self.opt_tours[
                self.BATCH_IDX[:, half_sub_tours_num:], self.first_node_idx[:,
                                                        half_sub_tours_num:] - length_of_sub_tour]
            next_opt_node = torch.cat((next_opt_node1, next_opt_node2), dim=-1)[:,:,None]
        next_edge_len = self.dist[self.BATCH_IDX.unsqueeze(-1), current_node.unsqueeze(-1), next_opt_node].mean(dim=-1)
        self.step_state = Step_State(first_node=first_node,current_node=current_node, ninf_mask=ninf_mask,
                                     next_opt_node=next_opt_node,
                                     next_edge_len=next_edge_len,
                                     length_of_sub_tour=length_of_sub_tour,
                                     sub_tours_size=sub_tours_size,
                                     BATCH_IDX=self.BATCH_IDX,
                                     first_node_idx=self.first_node_idx,
                                     problem_size=self.problem_size,
                                     opt_tours=self.opt_tours)
        return self.step_state

    def get_opt_prob(self):
        # In training phase
        sub_tours_size = self.problem_size * 2
        length_of_sub_tour = self.step_state.length_of_sub_tour
        if length_of_sub_tour == 1:
            probs_opt = torch.zeros(size=(self.batch_size, self.problem_size, self.problem_size))
            scatter_index1 = self.opt_tours[
                self.BATCH_IDX[:, :self.problem_size], (self.first_node_idx[:, :self.problem_size] + length_of_sub_tour) % self.problem_size]
            scatter_index2 = self.opt_tours[
                self.BATCH_IDX[:, self.problem_size:], self.first_node_idx[:, :self.problem_size] - length_of_sub_tour]
            scatter_index = torch.stack((scatter_index1, scatter_index2), dim=-1)
            probs_opt = probs_opt.scatter(dim=-1, index=scatter_index,
                                          src=torch.tensor([1.]).expand(scatter_index.shape))
            probs_opt = probs_opt.repeat(1,2,1) / 2.
        else:
            probs_opt = torch.zeros(size=(self.batch_size, sub_tours_size, self.problem_size))
            scatter_index1 = self.opt_tours[self.BATCH_IDX[:, :self.problem_size], (self.first_node_idx[:, :self.problem_size] + length_of_sub_tour) % self.problem_size]
            scatter_index2 = self.opt_tours[self.BATCH_IDX[:, self.problem_size:], self.first_node_idx[:, :self.problem_size] - length_of_sub_tour]
            scatter_index = torch.cat((scatter_index1,scatter_index2), dim=1)
            probs_opt = probs_opt.scatter(dim=-1,index=scatter_index[:,:,None],src=torch.tensor([1.]).expand(scatter_index[:,:,None].shape))

        return probs_opt

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def reset(self, test=False, pomo_size=100):
        if test:
            self.pomo_size = pomo_size
            self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
            self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
            self.selected_count = 0
            self.current_node = None
            # shape: (batch, pomo)
            self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
            # shape: (batch, pomo, 0~problem)

            # CREATE STEP STATE
            self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
            self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
            # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(self.problems, self.data_knn, self.knn_idx), reward, done


    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

