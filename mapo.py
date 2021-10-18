import numpy as np
import random
from numpy.core.fromnumeric import clip
from numpy.lib.shape_base import vsplit
import tensorflow as tf
from tensorflow.python.ops.gen_linalg_ops import self_adjoint_eig_eager_fallback
import tf_util as U

from distributions import make_FFpdtype
import tensorflow.contrib.layers as layers
from tensorflow.python.ops.gen_math_ops import log, tanh

def Agent_train(obs_shape_n, act_space_n, p_index, p_func, v_func, optimizer, clip_range=0.2, grad_norm_clipping=0.5, scope="trainer", reuse=None):
    '''
    List: obs_shape_n,    -the obsevation shape of each agent e.g. obs_shape_n[i] = 1, it shows the obseration shape of agent i is 1
    List: act_space_n,    -the action space of each agent, including the action shape, e.g. act_space_n[i] = [action1_limit, action2_limit, ...], where action1_limit = [max, min]
    Int: p_index,         -the ID of current agent
    Fun: p_func, v_func,  -the funcion of policy and value networks 
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # creat distributions
        act_pdtype = make_FFpdtype(act_space_n[p_index])

        # set up placeholders
        A = tf.placeholder(tf.float32,[None, 1], name="action"+str(p_index))
        X = tf.placeholder(tf.float32, [None, 3], name="observation"+str(p_index))
        ADV = tf.placeholder(tf.float32, [None], name="advantage"+str(p_index))
        R = tf.placeholder(tf.float32, [None], name="return"+str(p_index))
        OLDLOGPAC = tf.placeholder(tf.float32, [None], name="oldlogpac"+str(p_index))
        OLDVPRED = tf.placeholder(tf.float32, [None], name="oldvalue"+str(p_index))

        # policy (actor): from X to action distribution
        p = p_func(X, int(act_pdtype.param_shape()[0]), scope="p_func", num_units=[32, 8])
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype.pdfromflat(p)

        # log(probability) of A under current action distribution
        log_pac = tf.clip_by_value(act_pd.logp(A), -20.0, 0.0)

        # calculate ratio (pi(A|S) current policy / pi_old(A|S) old policy)
        ratio = tf.exp(log_pac - OLDLOGPAC)

        # defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range) 
        
        # final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        p_optimize_expr = U.minimize_and_clip(optimizer, pg_loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        p_train = U.function(inputs=[X] + [A] + [OLDLOGPAC] + [ADV], outputs=pg_loss, updates=[p_optimize_expr])
        log_px = U.function(inputs=[X] + [A], outputs=log_pac)

        # value (critic): from X to V(X)
        vpred = v_func(X, 1, scope="v_func", num_units=[32, 8])[:,0]
        v_func_vars = U.scope_vars(U.absolute_scope_name("v_func"))
        vpredclipped = OLDVPRED + tf.clip_by_value(vpred - OLDVPRED, - clip_range, clip_range)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        v_optimize_expr = U.minimize_and_clip(optimizer, vf_loss, v_func_vars, grad_norm_clipping)

        v_train = U.function(inputs=[X] + [OLDVPRED] + [R], outputs=vf_loss, updates=[v_optimize_expr])
        vs      = U.function(inputs=[X], outputs=vpred)

        return p_train, log_px, v_train, vs

def Agents_train(obs_shape_n, v_func, optimizer, clip_range=0.2, grad_norm_clipping=0.5, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        R = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        obs_ph_n = []
        for i in range(len(obs_shape_n)):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        v_input = tf.concat(obs_ph_n, 1)
        vpred = v_func(v_input, 1, scope="v_func", num_units=[64, 32])[:, 0]
        v_func_vars = U.scope_vars(U.absolute_scope_name("v_func"))
        vpredclipped = OLDVPRED + tf.clip_by_value(vpred - OLDVPRED, - clip_range, clip_range)

        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        v_optimize_expr = U.minimize_and_clip(optimizer, vf_loss, v_func_vars, grad_norm_clipping)

        v_train = U.function(inputs=obs_ph_n + [OLDVPRED] + [R], outputs=vf_loss, updates=[v_optimize_expr])
        vs      = U.function(inputs=obs_ph_n, outputs=vpred)

    return v_train, vs

class MAPOAgentTrainer(object):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, clip_range, args):
        self.p_train, self.log_px, self.v_train, self.vs = Agent_train(
            scope=name, 
            obs_shape_n=obs_shape_n, 
            act_space_n=act_space_n, 
            p_index=agent_index, 
            p_func=model, 
            v_func=model, 
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            clip_range=clip_range, 
            grad_norm_clipping=0.5, 
            reuse=tf.AUTO_REUSE)
    # optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
        self.args = args

    def log_p(self, obs, act):
        return self.log_px(*([obs] + [act]))

    def value_s(self, obs):
        return self.vs(*([obs]))
    
    def experience(self, num_raw_sample, raw_samples):

        self.obs_t, self.obs_tp1, self.act, self.rew, self.ret, self.val_st, self.val_stp1, self.adv, self.logp = [], [], [], [], [], [], [], [], []

         # deliver the samples
        self.obs_t      = raw_samples[0]
        self.act        = raw_samples[1]
        self.rew        = raw_samples[2]
        self.obs_tp1    = raw_samples[3]
        
        # compute the value of samples
        self.val_st     = self.value_s(self.obs_t)
        self.logp       = self.log_p(self.obs_t, self.act)
        self.val_stp1   = self.value_s(self.obs_tp1)

    def advantage(self, weight, group_adv):

        last_gae = 0
        for i in reversed(range(len(group_adv))):
            # TD difference of state_i
            delta = self.rew[i] + self.args.gamma * self.val_stp1[i]- self.val_st[i]
            # discounted gae of state_i    
            last_gae = delta + self.args.gamma * self.args.lam * last_gae
            # combine the individual gae (advantage) with the group gae
            com_gae = (1 - weight) * last_gae + weight * group_adv[i]
            self.adv.append(com_gae)
            # return at state_i
            self.ret.append(last_gae + self.val_st[i])

        # reverse the list
        self.adv.reverse()
        self.ret.reverse()

    def update(self, num_batch):

        p_loss, v_loss = [], []

        # compute the number of batch used to train
        num_samples = len(self.ret)

        # divide samples into n batches to train
        for start in range(0, num_samples, num_batch):
            end = start + num_batch
            if (end <= num_samples):
                obs_t = self.obs_t[start:end]
                act = self.act[start:end]
                logp = self.logp[start:end]
                adv = self.adv[start:end]
                ret = self.ret[start:end]
                p_loss.append(self.p_train(*([obs_t] + [act] + [logp] + [adv])))
                v_loss.append(self.v_train(*([obs_t] + [logp] + [ret])))
                
        return [p_loss, v_loss]

class MAPOAgentsTrainer(object):
    def __init__(self, name, model, obs_shape_n, clip_range, args):
        self.v_train, self.vs = Agents_train(
            scope=name, 
            obs_shape_n=obs_shape_n, 
            v_func=model, 
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr), 
            clip_range=clip_range,
            grad_norm_clipping=0.5, 
            reuse=False)
        self.args = args
    
    def value_s(self, obs):
        return self.vs(*(obs))
    
    def experience(self, num_raw_sample, raw_samples):

        self.obs_t, self.obs_tp1, self.act, self.rew, self.ret, self.val_st, self.val_stp1, self.adv = [], [], [], [], [], [], [], []

        # deliver the samples
        self.obs_t      = raw_samples[0]
        self.act        = raw_samples[1]
        self.rew        = raw_samples[2]
        self.obs_tp1    = raw_samples[3]
        
        # compute the value of states
        self.val_st     = self.value_s(self.obs_t)
        self.val_stp1   = self.value_s(self.obs_tp1)

        # cpmpute the return and advantage of states 
        last_gae = 0
        for i in reversed(range(num_raw_sample)):
            # TD difference at state_i
            delta = self.rew[i] + self.args.gamma * self.val_stp1[i]- self.val_st[i]
            # gae advantage at state_i
            last_gae = delta + self.args.gamma * self.args.lam * last_gae
            self.adv.append(last_gae)
            # return at state_i
            self.ret.append(self.adv[-1] + self.val_st[i])
        self.adv.reverse()
        self.ret.reverse()

        return self.adv
    
    def update(self, num_batch):

        v_loss = []

        # compute the number of batch used to train
        num_samples = len(self.ret)

        # divide samples into n batches to train
        for start in range(0, num_samples, num_batch):
            end = start + num_batch
            if (end <= num_samples):
                obs_t = [self.obs_t[i][start:end] for i in range(len(self.obs_t))]
                val = self.val_st[start:end]
                ret = self.ret[start:end]
                v_loss.append(self.v_train(*(obs_t + [val] + [ret])))

        return v_loss

# def mlp_model(input, num_outputs, scope, reuse=False, num_units=[32, 8], rnn_cell=None):
#     with tf.variable_scope(scope, reuse=reuse):
#         out = input
#         out = layers.fully_connected(out, num_outputs=num_units[0], activation_fn=tanh)
#         out = layers.fully_connected(out, num_outputs=num_units[1], activation_fn=tanh)
#         out = layers.fully_connected(out, num_outputs=num_outputs,  activation_fn=tanh)

#         return out

# def learn(arg):
#     with tf.Session() as sess:

#         agent = MAPOAgentTrainer(name="agent_0", model=mlp_model, obs_shape_n=[1], act_space_n=[[[0, 15000]]], agent_index=0, clip_range=0.2, args=arg)

#         sess.run(tf.global_variables_initializer())

#         obs = np.array([[1.0], [2.0]])
#         action = np.array([[3.0], [3.0]])
#         returns = np.array([[100.1], [4.0]])
#         value = np.array([[10.1], [5.0]])
#         advantage = np.array([[5.1], [6.0]])
#         log_pac = np.array([[0.1], [7.0]])
#         loss = agent.update(obs, action, returns, value, advantage, log_pac)
#         print(loss)

# if __name__ == "__main__":
#     arg = None
#     learn(arg)
