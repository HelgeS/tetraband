from __future__ import print_function

import argparse
import json
import os
import time
from datetime import datetime

import gym
from tqdm import tqdm

import pandas as pd

import agent
import envs
import feature_extractor
import random
import numpy as np


def main(args):
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    if args.save_dir and not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    basename = envs.BASENAMES[args.environment]

    env_name = '{basename}-{scenario}-{dataset}-v0'.format(
        basename=basename,
        scenario=args.scenario,
        dataset=args.dataset
    )
    env = gym.make(env_name)

    print("Environment {}".format(args.environment))
    print(env.observation_space)
    print(env.action_space)

    if args.features == 'net':
        extractor = feature_extractor.NetFeatureExtractor()
    elif args.features == 'hash':
        extractor = feature_extractor.ImageHashExtractor()
    else:
        raise NotImplementedError("Unknown feature extraction method '{}'".format(args.features))

    if args.agent == "baseline":
        # Special case -> handled somewhere else
        baseline(env, args)
        return
    elif args.agent == "bandit":
        actor = agent.BanditAgent(env, extractor)  # VW
    elif args.agent == "random":
        actor = agent.RandomAgent(env)

    obs = env.reset()
    print(obs)

    original_score = 0.0
    modified_score = 0.0
    totalreward = 0.0
    totalsuccess = 0.0
    iter_duration = 0.0
    statistics = {}

    for idx in range(env.action_space.n):
        statistics[env.actions[idx][0]] = {
            'action': env.actions[idx][0],
            'count': 0,
            'reward': 0.0,
            'success': 0
        }

        if env.is_hierarchical_action(idx):
            params = env.actions[idx][1]
            for param_idx in range(env.hierarchical_actions[idx]['space'].n):
                statistics[params[param_idx][0]] = {
                    'action': params[param_idx][0],
                    'count': 0,
                    'reward': 0.0,
                    'success': 0
                }

    log_file = '{timestamp}-{env}-{sc}-{agent}.json'.format(
        timestamp=datetime.now().strftime("%Y%m%d%H%M%S"),
        env=args.environment,
        sc=args.scenario,
        agent=args.agent)
    log_file = os.path.join(args.log_dir, log_file)

    for iteration in range(1, args.iterations + 1):
        start = time.time()
        act = actor.act(obs)
        obs, reward, done, info = env.step(act)

        actor.update(reward, done=done)
        iter_duration += time.time() - start

        action_name, param_name = env.get_action_name(act[0], act[1])

        statistics[action_name]['count'] += 1
        statistics[action_name]['reward'] += reward[0]
        statistics[action_name]['success'] += reward[0] > 0

        if param_name:
            statistics[param_name]['count'] += 1
            statistics[param_name]['reward'] += reward[1]
            statistics[param_name]['success'] += reward[1] > 0

        original_score += info['original_score']
        modified_score += info['modified_score']
        totalreward += reward[0]
        totalsuccess += reward[0] > 0

        if done:
            obs = env.reset()

        if (iteration % args.log_interval == 0) or iteration == args.iterations:
            stat_string = ' | '.join(
                ["{:.2f} ({:.2f}/{:d})".format(v['success'] / (v['count'] + 1e-10), v['success'],
                                               v['count']) for v in
                 statistics.values()])
            print("i = {}".format(iteration), round(totalsuccess / iteration, 2),
                  round(original_score / iteration, 2),
                  round(modified_score / iteration, 2), '\t', stat_string)

            log_dict = {
                'env': args.environment,
                'scenario': args.scenario,
                'agent': args.agent,
                'iteration': iteration,
                'totalreward': totalreward,
                'success': totalsuccess,
                'statistics': statistics,
                'original_accuracy': float(original_score) / iteration,
                'modified_accuracy': float(modified_score) / iteration,
                'duration': iter_duration / iteration
            }
            open(log_file, 'a').write(json.dumps(log_dict) + os.linesep)


def baseline(env, args):
    export_file = "logs/baseline_{}_{}_{}.csv".format(args.environment, args.scenario, args.dataset)
    env.random_images = False

    env.reset()

    if args.baseline_continue:
        file_mod = 'a'

        bl = pd.read_csv(export_file, sep=';')
        exist_imgs = set(bl.iloc[:, 0].unique())
        all_imgs = set(env.indices)
        image_indices = sorted(all_imgs - exist_imgs)
    else:
        file_mod = 'w'
        image_indices = sorted(env.indices)

    with open(export_file, file_mod) as f:
        if not args.baseline_continue:
            if args.environment == 'classification':
                print("image_id;action;parameter;action_reward;parameter_reward;success;original_score;modified_score;"
                      "original;prediction;label",
                      file=f)
            else:
                print("image_id;action;parameter;action_reward;parameter_reward;success;original_score;modified_score",
                      file=f)

        for image_idx in tqdm(image_indices):
            env.cur_image_idx = image_idx
            results = env.run_all_actions()

            for res in results:
                if args.environment == 'classification':
                    print("{};{};{};{};{};{};{};{};{};{};{}".format(env.cur_image_idx,
                                                                    res['action'],
                                                                    res['parameter'],
                                                                    res['action_reward'],
                                                                    res['parameter_reward'],
                                                                    res['success'],
                                                                    res['original_score'],
                                                                    res['modified_score'],
                                                                    res['original'],
                                                                    res['prediction'],
                                                                    res['label']), file=f)
                else:
                    print("{};{};{};{};{};{};{};{}".format(env.cur_image_idx,
                                                           res['action'],
                                                           res['parameter'],
                                                           res['action_reward'],
                                                           res['parameter_reward'],
                                                           res['success'],
                                                           res['original_score'],
                                                           res['modified_score']), file=f)
            f.flush()
            env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', default="classification",
                        choices=["classification", "detection"],
                        help="The SUT under test (classification: ResNet-34 for CIFAR-10, ResNet-50 for ImageNet, detection: Object detection API)")
    parser.add_argument('--scenario', default='basic', choices=['basic', 'rotation', 'hierarchical', 'shear'],
                        help='Test basic MRs or identify rotation robustness.')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'imagenet'],
                        help="Data set for image classification task (detection is always coco).")
    parser.add_argument('--agent', default="bandit", choices=["bandit", "random", "baseline"],
                        help="The MT selection agent (bandit: Contextual bandit, random: Selects random actions, baseline: Calculates the effect of every action on every state - VERY time expensive)")
    parser.add_argument('--features', default='net', choices=['net', 'hash'],
                        help="Feature extraction method (net: features from pretrained image classification model, hash: imagehash (pHash))")
    parser.add_argument('--iterations', default=1000, type=int,
                        help="How many iterations to run the testing")
    parser.add_argument('--save_dir', default=False, action='store_true',
                        help="Path to store agent model")
    parser.add_argument('--load_from', default=None,
                        help="Path to stored agent model (must fit --agent choice).")
    parser.add_argument('--log_dir', default='logs/', help="Directory to store log files")
    parser.add_argument('--predict', default=False,
                        help="Prediction mode, do not train the agent from feedback")
    parser.add_argument('--log_interval', type=int, default=10,
                        help="Number of iterations after which information is logged and printed")
    parser.add_argument('--baseline_continue', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=None, help="Random seed for number generators")
    args = parser.parse_args()

    if args.environment == 'detection':
        args.dataset = 'coco'

    main(args)
