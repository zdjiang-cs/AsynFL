import asyncio
import functools
import socket
import pickle
import os
from functools import singledispatch
import asyncio
from queue import Queue

import numpy as np
import torch
import torch.nn.functional as F
import copy
import time
import random

from config import *
from communication_module.comm_utils import *
from training_module import datasets, models, utils
from training_module.action import ServerAction


parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--model_type', type=str, default='AlexNet')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_pattern', type=float, default=0)
parser.add_argument('--alpha', type=float, default=200)
parser.add_argument('--algorithm', type=str, default='proposed')
parser.add_argument('--mode', type=str, default='adaptive')
parser.add_argument('--ratio', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--decay_rate', type=float, default=0.97)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--local_updates', type=int, default=50)
parser.add_argument('--time_budget', type=float, default=50000)
parser.add_argument('--heter', type=str, default='medium')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SERVER_IP = "127.0.0.1"


def scheduler(choose_num, prepare_time, current_lost_time):
	current_lost_time_order = sorted(current_lost_time.items(),key=lambda x:x[1],reverse=False)
	choose_workers = []

	for i in range(0,choose_num):
		choose_worker = current_lost_time_order[i][0]
		iteration_time = current_lost_time_order[i][1]
		choose_workers.append(choose_worker)
	iteration_time = current_lost_time_order[choose_num-1][1]

	for worker in list(current_lost_time.keys()):
		if current_lost_time[worker] <= iteration_time:
			current_lost_time[worker] = prepare_time[worker]
		else:
			current_lost_time[worker] = current_lost_time[worker] - iteration_time

	return choose_workers, current_lost_time, iteration_time

def aggregate_model(server_model, update_model, yita):
	params1 = server_model.state_dict().copy()
	params2 = update_model.state_dict().copy()
	with torch.no_grad():
		for name1 in params1:
			if name1 in params2:
				params1[name1] = ( 1 - yita ) * params1[name1] + yita * params2[name1]
	model = copy.deepcopy(server_model)
	model.load_state_dict(params1, strict=False)
	return model


def main():
    common_config = CommonConfig()
    common_config.model_type = args.model_type
    common_config.dataset_type = args.dataset_type
    common_config.batch_size = args.batch_size
    common_config.epoch = args.epoch
    common_config.learn_rate = args.lr
    common_config.local_iters = args.local_updates
    device = torch.device("cuda" if common_config.use_cuda and torch.cuda.is_available() else "cpu")

    worker_idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    yita = 0.3

    prepare_time = {}
    tau = {}
    count = {}
    epoch_comp_time = list()
    epoch_comm_time = list()
    epoch_time = list()
    true_epoch_time = list()
    total_time = 0
    total_comp_time = 0
    total_comm_time = 0
    true_time = 0
    capacity = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


    client_port=[47001, 47002, 47003, 47004, 47005, 47006, 47007, 47008, 47009, 47010]
    master_port=[57001, 57002, 57003, 57004, 57005, 57006, 57007, 57008, 57009, 57010]


    for idx, worker_idx in enumerate(worker_idx_list):
        custom = dict()

        common_config.worker_list.append(
            Worker(config=ClientConfig(idx=worker_idx,
                                       master_ip_addr=socket.gethostbyname(socket.gethostname()),
                                       action=ClientAction.LOCAL_TRAINING,
                                       custom=custom),
                   ip_addr=WORKER_IP_LIST[idx],
                   master_port=master_port[idx],
                   client_port=client_port[idx],
                   common_config=common_config
                   )
        )


    worker_num = len(common_config.worker_list)
    choose_num = int(worker_num * yita)
    for i in range(worker_num):
        prepare_time[i] = round(random.uniform(16,18)*capacity[i] ,2)
        count[i] = 0
    current_lost_time = copy.deepcopy(prepare_time)

    global_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    global_model = global_model.to(device)
    vm_models = [global_model for vm_idx in range(len(common_config.worker_list))]
    init_para = dict(global_model.named_parameters())
    model_tuple = models.Net2Tuple(global_model)

    train_dataset, test_dataset = datasets.load_datasets(common_config.dataset_type)
    partition_sizes = [
                        [0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1],
                        [0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1],
                        [0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1],
                        [0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1],
                        [0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1],
                        [0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1],
                        [0.0,  0.2,  0.0,  0.2,  0.0,  0.2,  0.0,  0.2,  0.0,  0.2],
                        [0.0,  0.2,  0.0,  0.2,  0.0,  0.2,  0.0,  0.2,  0.0,  0.2],
                        [0.2,  0.0,  0.2,  0.0,  0.2,  0.0,  0.2,  0.0,  0.2,  0.0],
                        [0.2,  0.0,  0.2,  0.0,  0.2,  0.0,  0.2,  0.0,  0.2,  0.0],
                      ]
    train_data_partition = utils.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    test_data_partition = utils.LabelwisePartitioner(test_dataset, partition_sizes=partition_sizes)


    log_out = dict()

    for worker_idx, worker in enumerate(common_config.worker_list):
            worker.config.para = init_para
            worker.config.model = model_tuple
            worker.config.custom["dataset_type"] = common_config.dataset_type
            worker.config.custom["model_type"] = common_config.model_type
            worker.config.custom["batch_size"] = common_config.batch_size
            worker.config.custom["learn_rate"] = common_config.learn_rate
            worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)
            worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)
            worker.connection = connect_get_socket("127.0.0.1", worker.master_port)
            print(worker.config.idx, "Connection success!")

    # Create dataset instance
    test_loader = utils.create_dataloaders(test_dataset, batch_size=128, shuffle=False)

    # TODO: Add thread to listen new client

    global_para = dict(global_model.named_parameters())

    action_queue = Queue()

    # Or you can add all action ad once
    
    start_time = time.time()

    for epoch_idx in range(1, 1 + common_config.epoch):

        if epoch_idx != 1:
            for idx, worker in enumerate(common_config.worker_list):
                if idx in choose_workers:
                    vm_models[idx] = copy.deepcopy(global_model)
                    worker.config.custom["learn_rate"] = min(common_config.learn_rate / (worker_num * f[idx]),0.07)
                worker.config.para = dict(vm_models[idx].named_parameters())
                worker.config.model = models.Net2Tuple(vm_models[idx])
                worker.config.epoch_num = epoch_idx

        for i in range(worker_num):
            prepare_time[i] = round(random.uniform(16,18)*capacity[i],2)

        (choose_workers, current_lost_time, iteration_time) = scheduler(choose_num, prepare_time, current_lost_time)
        
        print("before send")
        action_queue.put(ServerAction.SEND_STATES)
        ServerAction().execute_action(action_queue.get(), common_config.worker_list)
        print("after send")

        print("before get")
        action_queue.put(ServerAction.GET_STATES)
        ServerAction().execute_action(action_queue.get(), common_config.worker_list)
        print("after get")


        with torch.no_grad():
            for idx, worker in enumerate(common_config.worker_list):
                if idx in choose_workers:
                    count[idx] += 1
                    vm_models[idx].load_state_dict(worker.config.para)

            count_total = sum(count.values())
            f = {}
            for choose_worker in choose_workers:
                f[choose_worker] = count[choose_worker] / count_total

            flag = True
            for vm_idx in range(worker_num):
                if vm_idx in choose_workers:
                    if flag:
                        update_model = copy.deepcopy(vm_models[vm_idx])
                        flag = False
                    else:
                        update_model = utils.add_model(update_model, vm_models[vm_idx])

            update_model = utils.scale_model(
                update_model, 1.0 / choose_num)
            global_model = aggregate_model(global_model, update_model, yita)
            del update_model

        global_model = global_model.to(device)
        global_model.eval()
        test_loss = 0.0
        correct = 0 

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                if common_config.dataset_type == 'FashionMNIST' or common_config.dataset_type == 'MNIST':
                    if common_config.model_type == 'LR':
                        data = data.squeeze(1) 
                        data = data.view(-1, 28 * 28)
                
                output = global_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(1, keepdim=True)
                batch_correct = pred.eq(target.view_as(pred)).sum().item()

                correct += batch_correct

        test_loss /= len(test_loader.dataset)
        test_accuracy = np.float(1.0 * correct / len(test_loader.dataset))

        total_time = total_time + iteration_time
        utils.printer('Test set: Epoch: {} Time: {:.2f} Average loss: {:.2f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch_idx, total_time, test_loss, correct, len(test_loader.dataset), 100. * test_accuracy), log_out["acc_loss"])

    for worker in common_config.worker_list:
        worker.connection.shutdown(2)
        worker.connection.close()

if __name__ == "__main__":
    main()
