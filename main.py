import random
import torch
import numpy as np
from time import time
from prettytable import PrettyTable
from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate_kgil import test
from utils.helper import early_stopping
from model import EnvGenerator, Recommender


n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_dict(train_entity_pairs, start, end, train_user_set):

    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set)).to(device)
    return feed_dict


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    
    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """define model"""
    augmenter = EnvGenerator(args.K, args.dim, args.dim, device)
    model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)
    
    """define optimizer"""
    rec_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    aug_optimizer = torch.optim.Adam(augmenter.parameters(), lr=args.lr)
    model.augmenter = augmenter

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    total_iter = len(train_cf) // args.batch_size
    get_print = total_iter // 20
    
    print("start training ...")
    for epoch in range(args.epoch):
        
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        model.train()

        train_cf_pairs = train_cf_pairs[index]
        total_outer_loss, total_inner_loss, s = 0, 0, 0
        train_s_t = time()
        iteration = 0
        
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_pairs, s, s + args.batch_size, user_dict['train_user_set'])
            rec_loss, inv_mean, inv_var = model(batch)
            outer_loss = rec_loss + args.lamda* (inv_mean + inv_var)
            inner_loss = - inv_var
            rec_optimizer.zero_grad()
            outer_loss.backward()
            rec_optimizer.step()
            total_outer_loss += outer_loss.item()

            if epoch % args.epup == 0 and iteration > 10 and iteration % args.itup == 0:
                rec_loss, inv_mean, inv_var = model(batch)
                inner_loss = - inv_var
                aug_optimizer.zero_grad()
                inner_loss.backward()
                aug_optimizer.step()
                total_inner_loss += inner_loss.item()

            s += args.batch_size
            iteration += 1
            if iteration % get_print == 0:
                print("Train epoch:[{}/{}], iter:[{}/{}], outer_loss:[{:.6f}], inner_loss:[{:.6f}], time:[{:.2f}] min"
                .format(epoch + 1, args.epoch, 
                        iteration, total_iter, 
                        outer_loss.item(), 
                        inner_loss.item(),
                        (time() - train_s_t) / 60))

        train_e_t = time()
        print("-" * 100)
        print("start evluation ...")
        print("-" * 100)
        test_s_t = time()
        ret = test(model, user_dict, n_params)
        test_e_t = time()
        epoch_outer_loss = total_outer_loss  / total_iter
        epoch_inner_loss = total_inner_loss  / total_iter
        train_time = (train_e_t - train_s_t) / 60
        test_time = (test_e_t - test_s_t) / 60
        recall_20 = ret['recall'][0]
        ndcg_20 = ret['ndcg'][0]
        print("-" * 100)
        print("Test epoch:[{}/{}], loss:[out:{:.6f}, in:{:.6f}], recall:[{:.4f}], ndcg:[{:.4f}] | train time:[{:.2f}] min, test time:[{:.2f}] min"
            .format(epoch + 1, args.epoch, 
                    epoch_outer_loss, epoch_inner_loss,
                    recall_20, ndcg_20,
                    train_time, test_time))
        print("-" * 100)
        
