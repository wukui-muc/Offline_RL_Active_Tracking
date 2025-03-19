import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import numpy as np
import os
import torch
from gym_unrealcv.envs.tracking.baseline import PoseTracker
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE
import random
from collections import defaultdict

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.count_steps = 0
        self.action = self.action_space.sample()

    def act(self, observation, keep_steps=10):
        self.count_steps += 1
        if self.count_steps > keep_steps:
            self.action = self.action_space.sample()
            self.count_steps = 0
        else:
            return self.action
        return self.action

    def reset(self):
        self.action = self.action_space.sample()
        self.count_steps = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument("-e", "--env_id", default='UnrealTrackGeneral-FlexibleRoom-ContinuousColor-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=1, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=10, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=100, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')

    args = parser.parse_args()
    env = gym.make(args.env_id)
    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    if args.monitor:
        env = monitor.DisplayWrapper(env)

    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    # env = configUE.ConfigUEWrapper(env, offscreen=False,resolution=(160,160))
    env = agents.NavAgents(env, mask_agent=False)

    env.seed(int(args.seed))

    episode_count = 100
    rewards = 0
    done = False
    Total_rewards = 0
    try:
        for eps in range(1, episode_count):

            image = []
            action = []
            reward = []
            info_list =[]
            obs = env.reset()
            agents_num = len(env.action_space)
            tracker = PoseTracker(env.action_space[0], 200, 0)  # TODO support multi trackers
            tracker_random = RandomAgent(env.action_space[0])
            count_step = 0
            t0 = time.time()
            agents_num = len(obs)
            C_rewards = np.zeros(agents_num)
            actions=[[0,0],[0,0]]
            flag=0

            env.unwrapped.reward_params['exp_distance'] = 250
            env.unwrapped.reward_params['exp_angle'] = 0
            while True:

                obs, rewards, done, info = env.step(actions)

                target_pose = env.unwrapped.unrealcv.get_obj_pose(
                    env.unwrapped.player_list[env.unwrapped.target_id])
                tracker_pose=env.unwrapped.unrealcv.get_obj_pose(
                    env.unwrapped.player_list[env.unwrapped.tracker_id])

                actions = [np.array(tracker.act(tracker_pose, target_pose))]

                flag -= 1
                if random.random() < 0.2 or flag > 0:
                    actions[0] = tracker_random.act(obs,keep_steps=3)
                    if flag <= 0:
                        action_tmp = actions[0]
                        flag = random.randint(2, 3)
                    else:
                        actions[0] = action_tmp

                image.append(obs[env.unwrapped.tracker_id])
                action.append([actions[0]])
                reward.append(rewards)
                info_list.append(info)
                C_rewards += rewards
                count_step += 1

                cv2.imshow('rgb', (obs[0][:, :, 0:3].astype(np.uint8)))
                cv2.waitKey(1)
                if done:
                    fps = count_step/(time.time() - t0)
                    Total_rewards += C_rewards[0]
                    print ('Fps:' + str(fps), 'R:'+str(C_rewards), 'R_ave:'+str(Total_rewards/eps))
                    dict = {
                        'action': action,
                        'image': image,
                        'reward': reward,
                        'info': info_list,
                    }
                    save_dir = os.path.join(
                        'path/imperfect_' + "%04d" % int(eps) + "-%03d" % count_step + '.pt')
                    if count_step==500:
                        torch.save(dict, save_dir)
                    break

        # Close the env and write monitor result info to disk
        print('Finished')
        env.close()
    except KeyboardInterrupt:
        print('exiting')
        env.close()


