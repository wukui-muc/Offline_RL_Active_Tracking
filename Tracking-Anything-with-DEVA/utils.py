import time

import torch
import numpy as np
from deva import DEVAInferenceCore
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from deva.ext.grounding_dino import get_grounding_dino_model
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.result_utils import ResultSaver
from deva.ext.with_text_processor import process_frame_with_text as process_frame
import cv2
def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = '../trained_models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")

def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()
def collect_random_deva(env, dataset, num_samples=200, deva_model=None, deva_cfg=None, gd_model=None, sam_model=None):
    torch.autograd.set_grad_enabled(False)
    deva_cfg['temporal_setting'] = 'online'
    assert deva_cfg['temporal_setting'] in ['semionline', 'online', 'window']
    deva_cfg['enable_long_term_count_usage'] = True
    deva = DEVAInferenceCore(deva_model, config=deva_cfg)
    deva.next_voting_frame = deva_cfg['num_voting_frames'] - 1
    deva.enabled_long_id()
    result_saver = ResultSaver('./deva_out', None, dataset='demo', object_manager=deva.object_manager)

    state = env.reset()
    state_deva = process_frame(deva, gd_model, sam_model, str(0) + '.jpg', result_saver, 0,
                                    image_np=state[0][:, :, 0:3].astype(np.uint8))
    state = state_deva
    # state = torch.from_numpy(cv2.resize(state.astype(np.float32), (64, 64)).transpose(2, 0, 1)).float().cuda()
    state = cv2.resize(state.astype(np.float32), (64, 64)).transpose(2, 0, 1)

    for i in range(num_samples):
        action = env.action_space[0].sample()
        next_state, reward, done, _ = env.step([[action[0],action[1]]])
        next_state_deva = process_frame(deva, gd_model, sam_model, str(i+1) + '.jpg', result_saver, i+1,
                                        image_np=next_state[0][:, :, 0:3].astype(np.uint8))
        next_state = next_state_deva
        # next_state = torch.from_numpy(cv2.resize(next_state.astype(np.float32), (64, 64)).transpose(2, 0, 1)).float().cuda()
        next_state = cv2.resize(next_state.astype(np.float32), (64, 64)).transpose(2, 0, 1)

        dataset.add(state,
                    [[action[0],action[1]]], reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()
            state_deva = process_frame(deva, gd_model, sam_model, str(i+1) + '.jpg', result_saver, i+1,
                                       image_np=state[0][:, :, 0:3].astype(np.uint8))
            state = state_deva
            # state = torch.from_numpy(cv2.resize(state.astype(np.float32), (64, 64)).transpose(2, 0, 1)).float().cuda()
            state = cv2.resize(state.astype(np.float32), (64, 64)).transpose(2, 0, 1)

def evaluate(env, policy, eval_runs=5):
    """
    Makes an evaluation run with the current policy
    """
    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()

        rewards = 0
        while True:
            action = policy.get_action(state, eval=True)

            state, reward, done, _ = env.step(action)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)

def evaluate_deva(env, policy, eval_runs=1, deva_model=None, deva_cfg=None, gd_model=None, sam_model=None):
    print('start evaluate...')
    st_time = time.time()
    torch.autograd.set_grad_enabled(False)
    deva_cfg['temporal_setting'] = 'online'
    assert deva_cfg['temporal_setting'] in ['semionline', 'online', 'window']
    deva_cfg['enable_long_term_count_usage'] = True
    deva = DEVAInferenceCore(deva_model, config=deva_cfg)
    deva.next_voting_frame = deva_cfg['num_voting_frames'] - 1
    deva.enabled_long_id()
    result_saver = ResultSaver('./deva_out', None, dataset='demo', object_manager=deva.object_manager)
    min_val = np.array([-30, -100])  # need to be modified for different binary
    max_val = np.array([30, 100])
    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()
        state_deva = process_frame(deva, gd_model, sam_model, str(0) + '.jpg', result_saver, 0,
                                   image_np=state[0][:, :, 0:3].astype(np.uint8))
        state = state_deva
        # state = cv2.resize(state.astype(np.float32), (64, 64)).transpose(2, 0, 1)
        state = torch.from_numpy(cv2.resize(state.astype(np.float32), (64, 64)).transpose(2, 0, 1)).float().cuda()

        rewards = 0
        step=0
        while True:
            state = policy.CNN_Simple(state)
            action = policy.get_action(state.reshape(1,-1), eval=True)
            action = np.array(action)[0]  # 将数据从-1到1的范围反向归一化到0到1的范围
            denormalized_data = (action + 1) / 2

            # 将数据从0到1的范围反向归一化到原始范围
            denormalized_data = denormalized_data * (max_val - min_val) + min_val
            # need to be modified for different binary
            # action = [[denormalized_data[0][1], denormalized_data[0][0]]]
            action = [[denormalized_data[0], denormalized_data[1]]]

            state, reward, done, _ = env.step(action)
            state_deva = process_frame(deva, gd_model, sam_model, str(step) + '.jpg', result_saver, step,
                                       image_np=state[0][:, :, 0:3].astype(np.uint8))
            cv2.imshow('eval_deva',state_deva)
            cv2.waitKey(1)
            state = state_deva
            # next_state = np.concatenate([next_state_deva, next_state_depth], axis=2)
            state = torch.from_numpy(cv2.resize(state.astype(np.float32), (64, 64)).transpose(2, 0, 1)).float().cuda()
            # state = cv2.resize(state.astype(np.float32), (64, 64)).transpose(2, 0, 1)

            rewards += reward
            step+=1
            if done:
                break
        reward_batch.append(rewards)
    print('evaluate finished! cost time: ',time.time()-st_time)
    return np.mean(reward_batch)