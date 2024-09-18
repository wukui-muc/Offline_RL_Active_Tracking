import sys
import time
import numpy as np
import torch
import argparse

from absl.logging import exception

from utils import evaluate
import random
import gym
import os
import cv2
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation,configUE
torch.autograd.set_detect_anomaly = True
from agent_CNN_LSTM import CQLSAC_CNN_LSTM

torch.autograd.set_detect_anomaly = True
sys.path.append('/Tracking-Anything-with-DEVA/')
from deva import DEVAInferenceCore
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from deva.ext.grounding_dino import get_grounding_dino_model
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.result_utils import ResultSaver
from deva.ext.with_text_processor import process_frame_with_text as process_frame
def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="EvaluateModel", help="Run name, default: CQL-SAC")
    parser.add_argument("--env", type=str, default="UnrealTrack-demonstration_BUNKER-ContinuousColor-v0", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--max_distractor", type=int, default=2, help="")
    parser.add_argument("--load_agent_model", type=str, default='/home/wuk/CQL/CQL-SAC/trained_models/CQL-SAC-finetuned_v1CQL-SAC400.pth', help="")
    parser.add_argument("--input_type", type=str, default='Deva_cnn_lstm', help="")
    parser.add_argument("--seed", type=int, default=0, help="Seed, default: 1")
    parser.add_argument("--hidden_size", type=int, default=256, help="")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="")
    parser.add_argument("--temperature", type=float, default=1.0, help="")
    parser.add_argument("--cql_weight", type=float, default=1.0, help="")
    parser.add_argument("--target_action_gap", type=float, default=10, help="")
    parser.add_argument("--with_lagrange", type=int, default=0, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--eval_every", type=int, default=1, help="")
    parser.add_argument("--lstm_seq_len", type=int, default=20, help="")
    parser.add_argument("--lstm_out", type=int, default=64, help="")
    parser.add_argument("--lstm_layer", type=int, default=1, help="")


    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_text_default_args(parser)
    ##init deva
    deva_model, deva_cfg, args = get_model_and_config(parser)
    gd_model, sam_model = get_grounding_dino_model(deva_cfg, 'cuda')
    # args = parser.parse_args()
    return args ,deva_model, deva_cfg,gd_model, sam_model



def evaluate(env, agent, config,gd_model,sam_model,deva_cfg):

    next_state = env.reset()  # visualize the learning policy

    rewards = 0
    eval_steps = 0
    min_val = np.array([-30,-100]) #need to be modified for different binary
    max_val = np.array([30,100])
    ht = None
    ct = None

    torch.autograd.set_grad_enabled(False)
    deva_cfg['temporal_setting'] = 'online'
    assert deva_cfg['temporal_setting'] in ['semionline', 'online', 'window']
    deva_cfg['enable_long_term_count_usage'] = True
    deva = DEVAInferenceCore(deva_model, config=deva_cfg)
    deva.next_voting_frame = deva_cfg['num_voting_frames'] - 1
    deva.enabled_long_id()
    result_saver = ResultSaver('./deva_out', None, dataset='demo', object_manager=deva.object_manager)

    max_speed = random.randint(50, 60)
    # max_speed = 80
    env.unwrapped.unrealcv.set_max_speed(env.unwrapped.player_list[env.unwrapped.target_id], max_speed)

    while True:
        with torch.cuda.amp.autocast(enabled=deva_cfg['amp']):
            next_state_rgb= next_state[0][:,:,0:3]
            next_state_depth= np.expand_dims(next_state[0][:,:,-1],axis=2)
            next_state_deva=process_frame(deva, gd_model, sam_model, str(eval_steps)+'.jpg', result_saver, eval_steps, image_np=next_state_rgb.astype(np.uint8))
            # cv2.imshow('deva', next_state_deva.astype(np.uint8))

            # while eval_steps<1:
            #     target_mask=next_state_deva.sum(axis=2)
            #     target_mask = np.array(target_mask == 765)
            #     coords = np.where(target_mask != 0)
            #     # 计算坐标的最小和最大值来得到bounding box
            #     if np.size(coords)>0:
            #         y_min, x_min = np.min(coords[0]), np.min(coords[1])
            #         y_max, x_max = np.max(coords[0]), np.max(coords[1])
            #         bbox_area = (x_max - x_min) * (y_max - y_min)
            #         area_ratio_tmp = bbox_area / (target_mask.shape[0] * target_mask.shape[1])
            #         if area_ratio_tmp > 0.02 and area_ratio_tmp<0.2 :
            #             break
            #         else:
            #             cv2.imwrite('wrong_inital_deva_{}.jpg'.format(eval_steps),next_state_deva)
            #             cv2.imwrite('wrong_inital_{}.jpg'.format(eval_steps),next_state_rgb)
            #             raise exception('wrong initial')
            #     else:
            #         raise exception("no target")
                        # next_state=env.reset()
                        # rewards = 0
                        # eval_steps = 0
                        # result_saver = ResultSaver('./deva_out', None, dataset='demo',
                        #                            object_manager=deva.object_manager)
                        # next_state_rgb = next_state[0][:, :, 0:3]
                        # next_state_depth = np.expand_dims(next_state[0][:, :, -1], axis=2)
                        # next_state_deva = process_frame(deva, gd_model, sam_model, str(eval_steps) + '.jpg',
                        #                                 result_saver, eval_steps,
                        #                                 image_np=next_state_rgb.astype(np.uint8))



            if ('deva' in config.input_type.lower() or 'image' in config.input_type.lower()) and 'mlp' not in config.input_type.lower():
                next_state=next_state_deva
            if 'depth' in config.input_type.lower() or 'rgbd' in config.input_type.lower():
                next_state= np.concatenate([next_state_deva,next_state_depth],axis=2)
        with (torch.no_grad()):
            next_state = cv2.resize(next_state.astype(np.float32),(64,64)).transpose(2,0,1)
            next_state = torch.from_numpy(next_state).float().cuda().unsqueeze(0)
            if 'cnn' in config.input_type.lower() and 'lstm' in config.input_type.lower():
                next_state, ht, ct = agent.CNN_LSTM.inference(next_state.unsqueeze(0), ht, ct)
                # next_state = np.array(next_state.cpu())
                action = agent.get_action(next_state, eval=True)
                action=np.array(action)[0]
        assert len(action.shape)==2
        # 将数据从-1到1的范围反向归一化到0到1的范围
        denormalized_data = (action + 1) / 2

        # 将数据从0到1的范围反向归一化到原始范围
        denormalized_data = denormalized_data * (max_val - min_val) + min_val
        # need to be modified for different binary
        if 'SnowForest' in str(config.env):
            action = [[denormalized_data[0][1], denormalized_data[0][0]]]
        else:
            action = [[denormalized_data[0][0], denormalized_data[0][1]]]
        next_state, reward, done, info = env.step(action)
        # cv2.imshow('show', next_state[0][:,:,0:3].astype(np.uint8))
        # if eval_steps==49:
        #     print('eval_steps:', eval_steps)
        cv2.waitKey(1)
        rewards += reward
        eval_steps += 1
        if done:
            break
    return rewards, eval_steps


def eval_average(config, agent, env,gd_model,sam_model,deva_cfg):
    print('start evaluate...')
    AR = []
    EL = []
    start_time=time.time()
    print("Environment: ", config.env)
    while len(EL)<50:
        # try:
        reward, eval_steps = evaluate(env, agent, config,gd_model,sam_model,deva_cfg)
        if eval_steps>100:
            # print('episode：',len(EL),'reward:', reward, ' el:', eval_steps)
            # print('eval time: ', time.time() - start_time)

            AR.append(reward)
            EL.append(eval_steps)

        # except:
        #     pass
    AR_mean = sum(AR) / len(AR)
    AR_max = max(AR)
    AR_min = min(AR)
    EL_mean = sum(EL) / len(EL)
    EL_max = max(EL)
    EL_min = min(EL)
    print("AR：{},{},{}".format(AR_mean, AR_max - AR_mean, AR_min - AR_mean))
    print("EL：{},{},{}".format(EL_mean, EL_max - EL_mean, EL_min - EL_mean))
    EL_tmp = np.array(EL)
    print("success rate:{}".format(np.array([EL_tmp == 500]).sum() / len(EL_tmp)))
    print('total time: ', time.time()-start_time)

    AR_mean = sum(AR) / len(AR)
    AR_max = max(AR)
    AR_min = min(AR)
    EL_mean = sum(EL) / len(EL)
    EL_max = max(EL)
    EL_min = min(EL)
    print("AR：{},{},{}".format(AR_mean, AR_max - AR_mean, AR_min - AR_mean))
    print("EL：{},{},{}".format(EL_mean, EL_max - EL_mean, EL_min - EL_mean))
    EL_tmp = np.array(EL)
    print("success rate:{}".format(np.array([EL_tmp==500]).sum()/len(EL_tmp)))
    return AR_mean, EL_mean


def Eval_model(config,deva_model, deva_cfg, gd_model, sam_model):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    time_dilate = 10
    early_d = 50
    moni = False
    env = gym.make(config.env)
    if int(time_dilate) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, time_dilate)
    if int(early_d) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, early_d)
    if moni:
        env = monitor.DisplayWrapper(env)
    env.unwrapped.agents_type = ['player'] #['player', 'animal']
    if 'SnowForest' not in str(config.env):
        env = augmentation.RandomPopulationWrapper(env, config.max_distractor, config.max_distractor, random_target=False)
        env = configUE.ConfigUEWrapper(env, offscreen=True, resolution=(160, 160))
        env = agents.NavAgents(env, mask_agent=True)
    env.seed(config.seed)

    if 'deva' in config.input_type.lower() or 'image' in config.input_type.lower():
        if 'cnn' in config.input_type.lower() and 'lstm' in config.input_type.lower():
            agent = CQLSAC_CNN_LSTM(state_size=(3, 64, 64),
                                    action_size=2,
                                    tau=config.tau,
                                    hidden_size=config.hidden_size,
                                    learning_rate=config.learning_rate,
                                    temp=config.temperature,
                                    with_lagrange=config.with_lagrange,
                                    cql_weight=config.cql_weight,
                                    target_action_gap=config.target_action_gap,
                                    device=device,
                                    stack_frames=1,
                                    lstm_seq_len=config.lstm_seq_len,
                                    lstm_layer=config.lstm_layer,
                                    lstm_out=config.lstm_out)

    if 'devadepth' in config.input_type.lower() or 'rgbd' in config.input_type.lower():
        if 'cnn' in config.input_type.lower() and 'lstm' in config.input_type.lower():
            agent = CQLSAC_CNN_LSTM(state_size=(4, 64, 64),
                                    action_size=2,
                                    tau=config.tau,
                                    hidden_size=config.hidden_size,
                                    learning_rate=config.learning_rate,
                                    temp=config.temperature,
                                    with_lagrange=config.with_lagrange,
                                    cql_weight=config.cql_weight,
                                    target_action_gap=config.target_action_gap,
                                    device=device,
                                    stack_frames=1,
                                    lstm_seq_len=config.lstm_seq_len,
                                    lstm_layer=config.lstm_layer,
                                    lstm_out=config.lstm_out)



    assert config.load_agent_model !=None

    agent.load_state_dict(torch.load(os.path.join(config.load_agent_model)))
    print("load model: ",config.load_agent_model)

    AR_mean, EL_mean = eval_average(config, agent, env, gd_model, sam_model, deva_cfg)
    env.close()

if __name__ == "__main__":
    config, deva_model, deva_cfg, gd_model, sam_model = get_config()
    Eval_model(config,deva_model, deva_cfg, gd_model, sam_model)