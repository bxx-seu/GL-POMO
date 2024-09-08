import random as rd
import subprocess
import sys
import numpy as np
import pickle
import argparse
from datetime import datetime
import pytz

import os
sys.path.append(os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../../../')))
from problems.utils import read_tsplib

'''Based on the papers:
"Towards Feature-free TSP Solver Selection: A Deep Learning Approach"
"Evolving Diverse TSP Instances by Means of Novel and Creative Mutation Operators"

Portgen: A Graph Neural Network Assisted Monte Carlo Tree Search Approach to Traveling Salesman Problem

'''
from rpy2 import robjects as ro

def call_R_generate_data(operator, points_num, samples_num, seed):
    ro.r.source(os.path.join(os.path.split(os.path.realpath(__file__))[0], './tspgen_lib.R'))

    ro.r['operator'] = operator
    ro.r['points.num'] = points_num
    ro.r['ins.num'] = samples_num
    ro.r['seed'] = seed
    ro.r.source('./tspgen_func')

    x = ro.r['x']
    print(x)


def generate_multishape_data(distribution, graph_size=20, samples_num=1000):
    opts = ["explosion", "implosion", "cluster", "expansion", "rotation"]

    # generate rue instances
    if distribution == 'RUE':
        # Require the installation of instances generator of http://dimacs.rutgers.edu/archive/Challenges/TSP/
        rue_generator = './portgen/portgen'
        num_rue = samples_num

        dataset = []
        for i in range(num_rue):
            city_num = graph_size
            seed = rd.randint(1, 10000000)
            tmp_f = '%d.tsp' % (1)
            cmd = '%s %d %d > ./%s' %\
                  (rue_generator, city_num, seed, tmp_f)
            pid = subprocess.Popen(cmd, shell=True)
            pid.wait()
            nodes, _ = read_tsplib('./%s' % (tmp_f))
            assert len(nodes) == graph_size, f"len(nodes) != graph_size, {len(nodes)}"
            dataset.append(nodes)
        dataset = np.stack(dataset, axis=0)
    elif distribution == 'portcgen':
        # Require the installation of instances generator of http://dimacs.rutgers.edu/archive/Challenges/TSP/
        rue_generator = './portgen/portcgen'
        num_rue = samples_num

        dataset = []
        for i in range(num_rue):
            city_num = graph_size
            seed = rd.randint(1, 10000000)
            tmp_f = '%d.tsp' % (1)
            cmd = '%s %d %d > ./%s' %\
                  (rue_generator, city_num, seed, tmp_f)
            pid = subprocess.Popen(cmd, shell=True)
            pid.wait()
            nodes, _ = read_tsplib('./%s' % (tmp_f))
            assert len(nodes) == graph_size, f"len(nodes) != graph_size, {len(nodes)}"
            dataset.append(nodes)
        dataset = np.stack(dataset, axis=0)

    # generate netgen instances (centers from 4-8, 200)
    elif distribution == 'netgen':
        ro.r.source(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'netgen_lib.R'))
        seed = rd.randint(1, 1000000)

        # R
        ro.r.assign('points.num', graph_size)
        ro.r.assign('clu.lower', 2)
        ro.r.assign('clu.upper', 5)
        ro.r.assign('ins.num', samples_num)
        ro.r.assign('seed', seed)
        ro.r.source(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'netgen_func.R'))

        dataset = np.array(ro.r['dataset'])
        assert dataset.shape[-2] == graph_size, f"dataset.shape[-2] != graph_size, {dataset.shape[-2]}"

    # generate mutation instances
    elif distribution in opts:
        ro.r.source(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'tspgen_lib.R'))
        # num_mutation = 1000
        operator = distribution
        # f.write('%s Instances \n' % operator)
        seed = rd.randint(1, 1000000)

        # R
        ro.r.assign('operator', operator)
        ro.r.assign('points.num', graph_size)
        ro.r.assign('ins.num', samples_num)
        ro.r.assign('seed', seed)
        ro.r.source(os.path.join(os.path.split(os.path.realpath(__file__))[0],'tspgen_func.R'))

        dataset = np.array(ro.r['dataset'])
        assert dataset.shape[-2] == graph_size, f"dataset.shape[-2] != graph_size, {dataset.shape[-2]}"
    else:
        assert False, "Unknown distribution: {}".format(distribution)
    return dataset


def generate(task_name, distribution, graph_size=20, samples_num=1000, datadir='./data', log_f=None):
    coords_ = generate_multishape_data(distribution, graph_size=graph_size, samples_num=samples_num)

    # Rescale
    x_max, x_min = np.max(coords_, axis=(1,2), keepdims=True), np.min(coords_, axis=(1,2), keepdims=True)
    coords_norm=(coords_-x_min)/(x_max-x_min)
    coords_int = (coords_norm*10000000+0.5).astype(int)

    filedir = os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], datadir, '%s' % (task_name)))
    os.makedirs(os.path.join(filedir,'BigInt'), exist_ok=True)
    filename_int = os.path.join(os.path.join(filedir,'BigInt'), "BigInt-%s-n%d-%s-%d-classical.pkl" % (task_name, graph_size, distribution, samples_num))
    filename_unit = os.path.join(filedir, "unit-%s-n%d-%s-%d-classical.pkl" % (task_name, graph_size, distribution, samples_num))
    print(filename_int, 'dataset_shape: ', coords_.shape)
    if log_f:
        print(filename_int, 'dataset_shape: ', coords_.shape, file=log_f)
    pickle.dump(coords_int, open(filename_int,'wb'))
    pickle.dump(coords_norm, open(filename_unit, 'wb'))

# python generate_dataset_classical.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./data', help="Create datasets in data_dir/problem")
    parser.add_argument("--problem", type=str, default='tsp',
                        help="Problem, 'tsp', 'vrp', 'pctsp' or 'op_const', 'op_unif' or 'op_dist'"
                             " or 'all' to generate all")
    parser.add_argument('--Ns', type=int, nargs='+', default=[50, 60, 72, 87, 105, 126, 152, 183, 220],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument('--data_distribution', type=str, nargs='+', default='all',
                        help="Distributions to generate for problem, default 'all'.")
    parser.add_argument("--dataset_size", type=int, default=100, help="Size of the dataset")

    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=4321, help="Random seed")
    opts = parser.parse_args()

    def set_random(seed):
        rd.seed(seed)
        np.random.seed(seed)

    datadir = opts.data_dir
    task_name = opts.problem
    Ns = opts.Ns  # math.ceil(n+n*0.2)
    distributions1 = ["explosion", "implosion", "cluster", "expansion", "rotation"]
    distributions2 = ['RUE', 'portcgen', 'netgen']
    all_distri = distributions1 + distributions2 if opts.data_distribution == 'all' else opts.data_distribution
    samples_num = opts.dataset_size
    seed = opts.seed

    datasets_dir = os.path.join(os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], datadir)),
                                '%s' % (task_name))
    process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
    log = process_start_time.strftime("%Y%m%d_%H%M%S") + '-classical.log'

    with open(os.path.join(datasets_dir, log), 'w') as log_f:
        set_random(seed)
        print('Seed: ', seed, file=log_f)
        for N in Ns:
            for distri in all_distri:
                seed = rd.randint(1, 1000000)
                print('Distribution: ', distri, 'Seed: ', seed, file=log_f)
                set_random(seed)
                generate(task_name, distri, N, samples_num=samples_num, datadir=datadir, log_f=log_f)
