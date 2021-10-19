import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import subprocess

import tf_util as U 
from mapo import MAPOAgentTrainer, MAPOAgentsTrainer
import tensorflow.contrib.layers as layers
from WriteData import Write_MLP
from ReadData import Generate_Samples

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement learning for FastFarm")

    # Environment
    parser.add_argument("--num-episodes", type=int, default=200, help="number of episodes")

    # core training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="advantage estimation discounting factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=[64, 64], help="number of units in the mlp")

    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='Fast', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=10, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--plots-dir", type=str, default="learning_curves6/", help="directory where plot data is saved")

    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=[32, 8], rnn_cell=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units[0], activation_fn=tf.nn.sigmoid)
        out = layers.fully_connected(out, num_outputs=num_units[1], activation_fn=tf.nn.sigmoid)
        out = layers.fully_connected(out, num_outputs=num_outputs,  activation_fn=tf.nn.tanh)

        return out

def get_trainers(num_agent, obs_shape_n, act_space_n, clip_range, arglist):

    trainers = []
    model = mlp_model
    trainer = MAPOAgentTrainer
    for i in range(num_agent):
        trainers.append(trainer("agent_%d" % i, model, obs_shape_n, act_space_n, i, clip_range, arglist))

    # group trainer include the group value function 
    group_trainer = MAPOAgentsTrainer("group_trainer", model, obs_shape_n, clip_range, arglist)

    return trainers, group_trainer

def train(arglist):
    with U.single_threaded_session():

        # creat agent trainers
        num_agent = 3                                             # turbines in the fast farm
        obs_shape_n = [[3] for _ in range(num_agent)]               # the shape of observation (state) of each agent
        action_limit = [1, 14000]                                 # the output limit of policy network
        act_space_n = [[action_limit] for _ in range(num_agent)]    # action space of all agents
        clip_range = 0.2
        trainers, group_trainer = get_trainers(num_agent, obs_shape_n, act_space_n, clip_range, arglist)
        rated_power = 5.0e6

        # Initialize 
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == " ":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore:
            print('Loading previous state')
            U.load_state(arglist.load_dir)

        episode_rewards = []         # sum of reward for all agents
        agent_rewards   = [[] for i in range(num_agent)]            # individual agent rewards
        final_ep_rewards = []        # sum of reward for training curve
        final_ep_ag_rewards = [[] for i in range(num_agent)]        # agent rewards for training curve 
        saver = tf.train.Saver()
        train_step = 0
        write_file1 = "./fast-farm/Three_Turbines/WT1/spd_trq.dat"
        write_file2 = "./fast-farm/Three_Turbines/WT2/spd_trq.dat"
        write_file3 = "./fast-farm/Three_Turbines/WT3/spd_trq.dat"
        write_file = [write_file1, write_file2, write_file3]
        read_file1 = "./fast-farm/Three_Turbines/TSinflow.T1.out"
        read_file2 = "./fast-farm/Three_Turbines/TSinflow.T2.out"
        read_file3 = "./fast-farm/Three_Turbines/TSinflow.T3.out"
        # read_file1 = "/home/yubo/GBopenfast/openfast/reg_tests/r-test/glue-codes/fast-farm/TSinflow/TSinflow.T1.out"
        # read_file2 = "/home/yubo/GBopenfast/openfast/reg_tests/r-test/glue-codes/fast-farm/TSinflow/TSinflow.T2.out"
        # read_file3 = "/home/yubo/GBopenfast/openfast/reg_tests/r-test/glue-codes/fast-farm/TSinflow/TSinflow.T3.out"
        read_file = [read_file1, read_file2, read_file3]
        NumCol = 28
        ValidCol = [6, 26, 27, 25, 24]
        command = "./FAST.Farm ./fast-farm/Three_Turbines/TSinflow.fstf"
        t_start = time.time()

        print('Starting iterations...')
        for i in range(arglist.num_episodes):
            episode_rewards.append(0)
            agents_obs_t, agents_obs_tp1, agents_act, agents_rew  = [], [], [], 0

            # write policy network to OpenFAST input file
            for j in range(num_agent):
                scope = 'agent_%d'%j + '/p_func'
                variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)
                w = U.get_session().run(variables)
                writer = Write_MLP(write_file[j], w)
                print('Writing the policy network of agent {} to file: {}'.format(j, write_file[j]))
                writer.Write_File()

            # run FastFarm to collect training samples
            print('Running FAST.FARM...')
            error_message = subprocess.call(command, shell=True)
            if (error_message == 0):
                print('FAST.FARM is normally running')
            else:
                raise ValueError('There are some errors in FAST.FARM')

            # read the samples generated by FastFarm
            for j in range(num_agent):
                # agent_obs_t, agent_obs_tp1, agent_act, agent_rew = [], [], [], []
                reader = Generate_Samples(read_file[j], NumCol)
                reader.ReadData()
                num_sample, samples = reader.Data2Samples(ValidCol)
                if (num_sample < 1024):
                    raise ValueError('Samples generated by FAST.FARM are not enough')
                
                agent_obs_t   = samples[:, 0:3]
                agent_act     = np.expand_dims(samples[:, 3], axis=1)
                # if reward > rated power: reward = 0
                agent_rew     = samples[:, 4] * (samples[:, 4] <= rated_power)
                agent_rew     = np.maximum(agent_rew, 1.0)
                agent_rew[np.isnan(agent_rew)] = 1.0
                agent_obs_tp1 = samples[:, 5:8]
                agent_rewards[j].append(np.sum(agent_rew))
                episode_rewards[-1] += agent_rewards[j][-1]
                # print("agent_obs_t:")
                # print(agent_obs_t[0])
                # print("agent_act:")
                # print(agent_act[0])
                # print("agent_rew:")
                # print(agent_rew[0])
                # print("agent_obs_tp1:")
                # print(agent_obs_tp1[0])

                ############################
                # for k in range(num_sample):
                #     agent_obs_t.append([samples[k][0]])
                #     agent_act.append([samples[k][1]])
                #     agent_rew.append(samples[k][2])
                #     agent_obs_tp1.append([samples[k][3]])
                ############################

                agent_samples  = [agent_obs_t, agent_act, agent_rew, agent_obs_tp1]
                trainers[j].experience(num_sample, agent_samples)
                agents_obs_t.append(agent_obs_t)
                agents_act.append(agent_act)
                agents_rew += agent_rew
                agents_obs_tp1.append(agent_obs_tp1)


            agents_samples = [agents_obs_t, agents_act, agents_rew, agents_obs_tp1]
            group_adv = group_trainer.experience(num_sample, agents_samples)
            loss = group_trainer.update(arglist.batch_size)

            weight = float(i) / arglist.num_episodes
            for j in range(num_agent):
                trainers[j].advantage(weight, group_adv)
                loss = trainers[j].update(arglist.batch_size)
                # print("p_loss:")
                # print(loss[0])
                # print("v_loss:")
                # print(loss[1])
            
            train_step += num_sample // arglist.batch_size

            # save model, display training output
            if (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for j, rew in enumerate(agent_rewards):
                    final_ep_ag_rewards[j].append(np.mean(rew[-arglist.save_rate:]))
            
            if (i == arglist.num_episodes - 1):
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                group_file_name = arglist.plots_dir + arglist.exp_name + 'group_rewards.pkl'
                with open(group_file_name, 'wb') as fp:
                    pickle.dump(episode_rewards, fp)
                indiv_file_name = arglist.plots_dir + arglist.exp_name + 'indiv_rewards.pkl'
                with open(indiv_file_name, 'wb') as fp:
                    pickle.dump(agent_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
    
    
