import models
import argparse
from tqdm import tqdm
import torch.optim as optim
import networkx as nx
import torch
from torch.utils import data
import sys, os

import utils
from eval_metrics import *

import collections
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def evaluate(test_set, model, mode = 'test', use_gpu = 0):
    rec_list = {}
    actual_list = {}
    user_scores = {}
    # Evaluation is done for each user one at a time.
    # TODO: Optimize to evaluate batch-wise.
    for u, item_set in test_set.values:
        input_users = torch.LongTensor([u])
        if use_gpu:
            input_users = input_users.cuda()

        user_embed = model.user_embeddings(input_users)
        item_range = torch.arange(0, model.num_items)
        if use_gpu:
            item_range = item_range.cuda()
        scores = torch.mm(user_embed, model.item_embeddings(item_range).t()).squeeze()
        user_scores[u] = scores.detach().cpu().numpy()
        _, rec_list_u = scores.topk(100, dim=0)
        rec_list[u] = rec_list_u.cpu().numpy().tolist()
        actual_list[u] = dict(zip(item_set, item_set))
        #print (actual_list[u], rec_list[u], np.isin(rec_list[u], actual_list[u].keys()))

    rows_to_output_full = ranking_metrics(rec_list, actual_list, method ='bpr')
    print ("\n\n")
    for row in rows_to_output_full:
        if row['Metric'] == 'Recall' and row['K'] == 50:
            return row['score']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run script parameters')

    parser.add_argument('--dataset', type=str, nargs='?', default='yelp',
                        help='Dataset Name')

    parser.add_argument('--method', type=str, nargs='?', default='BPR',
                        help='method name')

    parser.add_argument('--random_split', type=int, nargs='?', default=1,
                        help='argument to specify random split')

    parser.add_argument('--embed_dim', type=int, nargs='?', default=64,
                        help='embedding dimension')

    parser.add_argument('--batch_size', type=int, nargs='?', default=1024,
                        help='batch size')

    parser.add_argument('--epochs', type=int, nargs='?', default=20,
                        help='# epochs')

    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.0001,
                        help='learning rate')

    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0001,
                        help='L2 reg weight decay')

    parser.add_argument('--lambda_u', type=float, nargs='?', default=0.0000,
                        help='L2 reg hyper-param (users)')

    parser.add_argument('--lambda_i', type=float, nargs='?', default=0.0000,
                        help='L2 reg hyper-param (items)')

    parser.add_argument('--val_iter', type=int, nargs='?', default=1,
                        help='Interval for validation')

    parser.add_argument('--use_gpu', type=int, nargs='?', default=0,
                        help='Flag to use GPU')

    args = parser.parse_args()
    SEED = 7
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_ratings, test_set, num_items, num_users = utils.load_data(args.dataset, args.random_split)

    train_users = [x[0] for x in train_ratings]
    train_users = {x:0 for x in train_users}
    # data loader params
    train_loader_params = {'batch_size': args.batch_size,
                           'shuffle': True,
                           'num_workers': 1}

    test_loader_params = {'batch_size': args.batch_size,
                          'shuffle': False,
                          'num_workers': 1}
    print ("# users", num_users)

    training_set = utils.Dataset(train_ratings)
    train_generator = data.DataLoader(training_set, **train_loader_params)

    model_params = {'num_users': num_users, 'num_items': num_items, 'embed_dim': args.embed_dim, 'method': args.method, 'lambda_u': args.lambda_u, 'lambda_i': args.lambda_i, 'use_gpu': args.use_gpu}

    model = models.BPR(**model_params)
    if args.use_gpu:
        model = model.cuda()

    if args.method == 'BPR':
       optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_performance = 0.0
    checkpoint_path = '../models/{}_{}.pt'.format(args.method, args.dataset)

    train = 0.0

    # Preprocessing
    user_restaurants = collections.defaultdict(set)
    fp = csv.reader(open("../data/train.csv"), delimiter=',')
    for i, data in enumerate(fp):
        if i != 0:
            user_restaurants[int(data[0])].add(int(data[1]))

    user_users = collections.defaultdict(set)
    fp = csv.reader(open("../data/social.csv"), delimiter=',')
    for i, data in enumerate(fp):
        if i != 0:
            user_users[int(data[0])].add(int(data[1]))
            user_users[int(data[1])].add(int(data[0]))

    restaurant_category = {}
    fp = csv.reader(open("../data/item_cats.csv"), delimiter=',')
    for i, data in enumerate(fp):
        if i != 0:
            restaurant_category[int(data[0])] = str(data[1])
    
    user_categories = collections.defaultdict(set)
    for user in user_restaurants:
        restaurants = user_restaurants[user]
        for restaurant in restaurants:
            user_categories[user].add(restaurant_category[restaurant])

    # compile 5 most similar users based on all 3 sets of data
    # most_similar_users = collections.defaultdict(list)
    # for user in user_categories:
    #     categories = user_categories[user]
    #     users = user_users[user]
    #     for other_user in users: # go through users they have interacted with
    #         other_categories = user_categories[other_user]
    #         if (len(categories.intersection(other_categories)) > 0): # if category preference is similar add to list
    #             if len(most_similar_users[user]) < 5:
    #                 most_similar_users[user].append(other_user)
    #     most_similar_users[user] += [user] * (5 - len(most_similar_users[user]))

    # Compute most similar users based on all 3 sets of data
    most_similar_user = {}
    for user in user_categories:
        categories = user_categories[user]
        users = user_users[user]
        most_similar = 0
        for other_user in users: # go through users they have interacted with
            other_categories = user_categories[other_user]
            if (len(categories.intersection(other_categories)) > most_similar): # if category preference is better
                most_similar_user[user] = other_user
                most_similar = len(categories.intersection(other_categories))
        if user not in most_similar_user:
            most_similar_user[user] = user

    for epoch in range(args.epochs):
        # Training
        model.train()
        epoch_loss = 0.
        for user_batch, item_batch in tqdm(train_generator):
            model.zero_grad()
            optimizer.zero_grad()
            user_batch = torch.LongTensor(user_batch)
            item_batch = torch.LongTensor(item_batch)
            similar_user_batch = torch.tensor([most_similar_user[user.item()] for user in user_batch])
            if args.use_gpu:
                user_batch = user_batch.cuda()
                item_batch = item_batch.cuda()
                similar_user_batch = similar_user_batch.cuda()
            pos_preds = model(user_batch, item_batch)
            pos_similar_user_preds = model(similar_user_batch, item_batch)
            new_pos_preds = pos_preds * (1.0 - train) + pos_similar_user_preds * train
            negative_items = model.sample_negs(item_batch, 1) # [B, 1]
            if args.use_gpu:
                negative_items = negative_items.cuda()
            neg_preds = model(user_batch, negative_items.view(-1))
            neg_similar_user_preds = model(similar_user_batch, negative_items.view(-1))
            new_neg_preds = neg_preds * (1.0 - train) + neg_similar_user_preds * train
            batch_loss = model.loss(new_pos_preds, new_neg_preds)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss
        epoch_loss /= len(train_generator)

        if epoch % args.val_iter == 0:
            train = min(0.25, train + 0.01)
            model.eval()
            val_performance = evaluate(test_set, model, use_gpu=args.use_gpu, mode = 'val')
            print ("val performance", val_performance, "best val performance : ", best_val_performance)

            if val_performance > best_val_performance:
                # print ("saving best model ... at ", val_performance)
                best_val_performance = val_performance
                torch.save(model.state_dict(), checkpoint_path)

        print ("Epoch {}: Mean Loss: {}".format(epoch, epoch_loss))

    model = models.BPR(**model_params)
    model.load_state_dict(torch.load(checkpoint_path))
    if args.use_gpu:
        model = model.cuda()
    model.eval()
    print ("Evaluating on test set: ")
    evaluate(test_set, model, use_gpu=args.use_gpu, mode = 'test')
