import torch.optim as optim
from my_model import *
from utils import *
import design_bench
import argparse
import os

from register_dataset.register_rosenbrock import RosenbrockDataset, RosenbrockOracle

design_bench.register('Rosenbrock-Exact-v0', RosenbrockDataset, RosenbrockOracle,
                      # keyword arguments for building the dataset
                      dataset_kwargs=dict(max_samples=None, distribution=None, max_percentile=50, min_percentile=0),
                      # keyword arguments for building the exact oracle
                      oracle_kwargs=dict(noise_std=0.0)
                      )
from register_dataset.register_rastrigin import RastriginDataset, RastriginOracle

design_bench.register('Rastrigin-Exact-v0', RastriginDataset, RastriginOracle,
                      # keyword arguments for building the dataset
                      dataset_kwargs=dict(max_samples=None, distribution=None, max_percentile=50, min_percentile=0),
                      # keyword arguments for building the exact oracle
                      oracle_kwargs=dict(noise_std=0.0)
                      )

from register_dataset.register_levy import LevyDataset, LevyOracle

design_bench.register('Levy-Exact-v0', LevyDataset, LevyOracle,
                      # keyword arguments for building the dataset
                      dataset_kwargs=dict(max_samples=None, distribution=None, max_percentile=50, min_percentile=0),
                      # keyword arguments for building the exact oracle
                      oracle_kwargs=dict(noise_std=0.0)
                      )

# from register_dataset.register_rnabind_random import RNABindDataset
# from design_bench.oracles.sklearn import RandomForestOracle
#
# design_bench.register('RNABind-RandomForest-v0', RNABindDataset, RandomForestOracle,
#                       # keyword arguments for building the dataset
#                       dataset_kwargs=dict(
#                           max_samples=None,
#                           distribution=None,
#                           max_percentile=50,
#                           min_percentile=0,
#                       ),
#
#                       # keyword arguments for building RandomForest oracle
#                       oracle_kwargs=dict(
#                           noise_std=0.0,
#                           max_samples=2000,
#                           distribution=None,
#                           max_percentile=100,
#                           min_percentile=0,
#
#                           # parameters used for building the model
#                           model_kwargs=dict(n_estimators=100,
#                                             max_depth=100,
#                                             max_features="auto"),
#                       ))
from register_dataset.register_rnabind_exact import RNABindDataset, RNABindOracle

design_bench.register('RNABind-Exact-v0', RNABindDataset, RNABindOracle,
                      # keyword arguments for building the dataset
                      dataset_kwargs=dict(max_samples=None, distribution=None, max_percentile=50, min_percentile=0),
                      # keyword arguments for building the exact oracle
                      oracle_kwargs=dict(noise_std=0.0)
                      )

device = torch.device('cuda:' + '0')
set_seed(123)


def train_proxy(args):
    if args.task != 'TFBind10-Exact-v011':
        task = design_bench.make(args.task)
    else:
        task = design_bench.make(args.task,
                                 dataset_kwargs={"max_samples": 30000})
    # task_y0 = task.y
    # task_x, task_y, length = process_data(task, args.task, task_y0)
    task_x, task_y, length = process_data_new(task, args.task)

    task_x = torch.Tensor(task_x).to(device)
    task_y = torch.Tensor(task_y).to(device)
    L = task_x.shape[0]
    indexs = torch.randperm(L)
    task_x = task_x[indexs]
    task_y = task_y[indexs]
    # print(labels_norm)
    # train_L = int(L * 0.90)
    train_L = L

    # # normalize labels
    # train_labels0 = task_y[0: train_L]
    # valid_labels = task_y[train_L:]
    # # load logits
    # train_logits0 = task_x[0: train_L]
    # valid_logits = task_x[train_L:]
    # T = int(train_L / args.bs) + 1

    # normalize labels
    train_labels0 = task_y
    valid_labels = task_y
    # load logits
    train_logits0 = task_x
    valid_logits = task_x
    T = int(train_L / args.bs) + 1

    # define model
    model = SimpleMLP(task_x.shape[1]).to(device)
    # opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    opt = optim.AdamW(model.parameters(),
                      lr=args.lr,
                      betas=(0.9, 0.999),
                      weight_decay=args.wd,
                      )
    # begin training
    best_pcc = -1
    for e in range(args.epochs):
        # adjust lr
        adjust_learning_rate(opt, args.lr, e, args.epochs)
        # random shuffle
        indexs = torch.randperm(train_L)
        train_logits = train_logits0[indexs]
        train_labels = train_labels0[indexs]
        tmp_loss = 0
        for t in range(T):
            x_batch = train_logits[t * args.bs:(t + 1) * args.bs, :]
            y_batch = train_labels[t * args.bs:(t + 1) * args.bs]
            pred = model(x_batch)
            loss = torch.mean(torch.pow(pred - y_batch, 2))
            tmp_loss = tmp_loss + loss.data
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            valid_preds = model(valid_logits)
        pcc = compute_pcc(valid_preds.squeeze(), valid_labels.squeeze())
        # valid_loss = torch.mean(torch.pow(valid_preds.squeeze() - valid_labels.squeeze(),2))
        # print("epoch {} training loss {} loss {} best loss {}".format(e, tmp_loss/T, valid_loss, best_val))
        print("\nepoch {} training loss {} pcc {} best pcc {}".format(e, tmp_loss / T, pcc, best_pcc))
        if pcc > best_pcc:
            best_pcc = pcc
            # print("epoch {} has the best loss {}".format(e, best_pcc))
            torch.save(model.state_dict(),
                       os.path.join(args.store_path, args.task + "_proxy_" + str(args.seed) + ".pt"))
            # print('pred', valid_preds[0:20])
    print('SEED', str(args.seed), 'has best pcc', str(best_pcc))


def design_opt(args):
    if args.task != 'TFBind10-Exact-v0':
        task = design_bench.make(args.task)
    else:
        task = design_bench.make(args.task,
                                 dataset_kwargs={"max_samples": 30000})
    # x = task.x[0:1]
    # y = task.y[0:1]
    # print(y, task.predict(x))
    # exit()
    # load_y(args.task)
    # task_y0 = task.y
    # task_x, task_y, length = process_data(task, args.task, task_y0)
    task_x, task_y, length = process_data_new(task, args.task)
    task_x = torch.Tensor(task_x).to(device)
    task_y = torch.Tensor(task_y).to(device)

    index = torch.argsort(-task_y.squeeze())
    args.topk = length
    index = index[:args.topk]
    x_init = copy.deepcopy(task_x[index])
    y_init = copy.deepcopy(task_y[index])
    gt_score_after_list = []
    pred_score_after_list = []
    new_design = []

    if args.method == 'simple':
        proxy = SimpleMLP(task_x.shape[1]).to(device)
        proxy.load_state_dict(
            torch.load(os.path.join(args.store_path, args.task + "_proxy_" + str(args.seed) + ".pt"),
                       map_location='cuda:0'))
    else:
        proxy1 = SimpleMLP(task_x.shape[1]).to(device)
        proxy1.load_state_dict(
            torch.load(os.path.join(args.store_path, args.task + "_proxy_" + str(args.seed) + ".pt"),
                       map_location='cuda:0'))
        proxy2 = SimpleMLP(task_x.shape[1]).to(device)
        proxy2.load_state_dict(
            torch.load(os.path.join(args.store_path, args.task + "_proxy_" + str(args.seed) + ".pt"),
                       map_location='cuda:0'))
        proxy3 = SimpleMLP(task_x.shape[1]).to(device)
        proxy3.load_state_dict(
            torch.load(os.path.join(args.store_path, args.task + "_proxy_" + str(args.seed) + ".pt"),
                       map_location='cuda:0'))

    for x_i in range(x_init.shape[0]):
        candidate = copy.deepcopy(x_init[x_i:x_i + 1])
        gt_score_before = task.predict(candidate.cpu().numpy().reshape(1, *task.x.shape[1:]))
        estimate_score_before = proxy(candidate)
        candidate.requires_grad = True
        candidate_opt = optim.Adam([candidate], lr=args.ft_lr)
        for i in range(1, args.Tmax + 1):
            if args.method == 'simple':
                loss = -proxy(candidate)
            elif args.method == 'ensemble':
                loss = -1.0 / 3.0 * (proxy1(candidate) + proxy2(candidate) + proxy3(candidate))
            candidate_opt.zero_grad()
            loss.backward()
            candidate_opt.step()
        gt_score_after = task.predict(candidate.cpu().detach().numpy().reshape(1, *task.x.shape[1:]))
        estimate_score_after = proxy(candidate)
        print(f"\nindex: {x_i}")
        # print(f"original design: {x_init[x_i]}")
        print(f"gt score before: {y_init[x_i]}")
        print(f"gt score before: {gt_score_before.squeeze()}")
        print(f"proxy score before: {estimate_score_before.squeeze().cpu().detach().numpy()}")
        # print(f"optimized design: {candidate.data}")
        print(f"gt score after: {gt_score_after.squeeze()}")
        print(f"proxy score after: {estimate_score_after.squeeze().cpu().detach().numpy()}")
        gt_score_after_list.append(gt_score_after.squeeze())
        pred_score_after_list.append(estimate_score_after.squeeze().cpu().detach().numpy())
        new_design.append(candidate.squeeze().cpu().detach().numpy())

    # save to dict adn store in file
    to_save = {}
    to_save["x"] = new_design
    to_save["gt_y"] = gt_score_after_list
    to_save["pred_y"] = pred_score_after_list
    # save to file with numpy
    np.save(os.path.join(args.store_path, args.task + "_pseudo_target_" + str(args.seed) + ".npy"), to_save)


def experiment():
    task = [args.task]
    seeds = list(range(9))

    # Training Proxy
    args.mode = 'train'
    for s in seeds:
        print("Current seed is " + str(s), end="\t")
        args.seed = s
        set_seed(args.seed)
        for t in task:
            print("Current task is " + str(t))
            args.task = t
            print("this is my setting", args)
            train_proxy(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pairwise offline")
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--task', choices=['Superconductor-RandomForest-v0', 'HopperController-Exact-v0',
                                           'AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0', 'TFBind8-Exact-v0',
                                           'CIFARNAS-Exact-v0', 'TFBind10-Exact-v0', 'Rosenbrock-Exact-v0',
                                           'Ackley-Exact-v0', 'Cosines-Exact-v0', 'Griewank-Exact-v0', 'Levy-Exact-v0',
                                           'Rastrigin-Exact-v0', 'Sphere-Exact-v0', 'Zakharov-Exact-v0',
                                           'RNABind-RandomForest-v0', 'RNABind-Exact-v0'],
                        type=str, default='CIFARNAS-Exact-v0')
    parser.add_argument('--mode', choices=['design', 'train'], type=str, default='design')
    # grad descent to train proxy
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--wd', default=0.0, type=float)
    # grad ascent to obtain design
    parser.add_argument('--Tmax', default=100, type=int)  # Usually 100 for discrete tasks and 200 for continuous tasks
    parser.add_argument('--ft_lr', default=1e-2,
                        type=float)  # Usually 1e-1 for discrete tasks and 1e-3 for continuous tasks
    parser.add_argument('--topk', default=1000, type=int)
    parser.add_argument('--interval', default=100, type=int)
    parser.add_argument('--method', choices=['ensemble', 'simple'], type=str, default='simple')
    parser.add_argument('--seed1', default=1, type=int)
    parser.add_argument('--seed2', default=10, type=int)
    parser.add_argument('--seed3', default=100, type=int)
    parser.add_argument('--store_path', default="generated_target_dist/", type=str)
    args = parser.parse_args()
    if args.mode == 'train':
        train_proxy(args)
    elif args.mode == 'design':
        design_opt(args)
