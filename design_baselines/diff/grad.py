import torch.optim as optim
from my_model import *
from utils import *
import design_bench
import argparse

device = torch.device('cuda:' + '0')


def train_proxy(args):
    task = design_bench.make(args.task)
    task_y0 = task.y
    task_x, task_y, shape0 = process_data(task, args.task, task_y0)

    task_x = torch.Tensor(task_x).to(device)
    task_y = torch.Tensor(task_y).to(device)
    L = task_x.shape[0]
    indexs = torch.randperm(L)
    task_x = task_x[indexs]
    task_y = task_y[indexs]
    # print(labels_norm)
    train_L = int(L * 0.90)
    # normalize labels
    train_labels0 = task_y[0: train_L]
    valid_labels = task_y[train_L:]
    # load logits
    train_logits0 = task_x[0: train_L]
    valid_logits = task_x[train_L:]
    T = int(train_L / args.bs) + 1
    # define model
    model = SimpleMLP(task_x.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
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
        print("epoch {} training loss {} pcc {} best pcc {}".format(e, tmp_loss / T, pcc, best_pcc))
        if pcc > best_pcc:
            best_pcc = pcc
            print("epoch {} has the best loss {}".format(e, best_pcc))
            torch.save(model.state_dict(), args.store_path + args.task + "_proxy_" + str(args.seed) + ".pt")
            print('pred', valid_preds[0:20])
    print('SEED', str(args.seed), 'has best pcc', str(best_pcc))


def design_opt(args):
    task = design_bench.make(args.task)
    load_y(args.task)
    task_y0 = task.y
    task_x, task_y, shape0 = process_data(task, args.task, task_y0)
    task_x = torch.Tensor(task_x).to(device)
    task_y = torch.Tensor(task_y).to(device)

    indexs = torch.argsort(task_y.squeeze())
    index = indexs[-args.topk:]
    x_init = copy.deepcopy(task_x[index])
    scores = []
    new_design = []
    for x_i in range(x_init.shape[0]):
        if args.method == 'simple':
            proxy = SimpleMLP(task_x.shape[1]).to(device)
            proxy.load_state_dict(
                torch.load(args.store_path + args.task + "_proxy_" + str(args.seed) + ".pt", map_location='cuda:0'))
        else:
            proxy1 = SimpleMLP(task_x.shape[1]).to(device)
            proxy1.load_state_dict(
                torch.load(args.store_path + args.task + "_proxy_" + str(args.seed1) + ".pt", map_location='cuda:0'))
            proxy2 = SimpleMLP(task_x.shape[1]).to(device)
            proxy2.load_state_dict(
                torch.load(args.store_path + args.task + "_proxy_" + str(args.seed2) + ".pt", map_location='cuda:0'))
            proxy3 = SimpleMLP(task_x.shape[1]).to(device)
            proxy3.load_state_dict(
                torch.load(args.store_path + args.task + "_proxy_" + str(args.seed3) + ".pt", map_location='cuda:0'))
        candidate = x_init[x_i:x_i + 1]
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
        estimate_score_after = proxy(candidate)
        print("candidate {} score before {} score now {}".format(x_i, estimate_score_before.squeeze(),
                                                                 estimate_score_after.squeeze()))
        scores.append(estimate_score_after.squeeze())
        new_design.append(candidate.data)
        x_init[x_i] = candidate.data
    # save to dict adn store in file
    to_save = {}
    to_save["x"] = new_design
    to_save["y"] = scores
    # save to file with numpy
    np.save(args.store_path + args.task + "_pseudo_target_" + str(args.seed) + ".npy", to_save)


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
                                           'CIFARNAS-Exact-v0', 'TFBind10-Exact-v0'],
                        type=str, default='CIFARNAS-Exact-v0')
    parser.add_argument('--mode', choices=['design', 'train'], type=str, default='design')
    # grad descent to train proxy
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--wd', default=0.0, type=float)
    # grad ascent to obtain design
    parser.add_argument('--Tmax', default=100, type=int)  # Usually 100 for discrete tasks and 200 for continuous tasks
    parser.add_argument('--ft_lr', default=1e-1, type=float)  # Usually 1e-1 for discrete tasks and 1e-3 for continuous tasks
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
