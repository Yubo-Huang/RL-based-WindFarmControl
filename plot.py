import pickle
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import scipy.stats as st 

def mean_and_std(x):
    '''
    Function: this function is used to compute the mean and std of x
    x: m*n array, the input data
    '''

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)

    return x_mean, x_std

def plot_results(x, y_mean, y_std, Label, Figname):
    '''
    Function: this function is used to plot the fast.farm results
    x: 1*n array, x-axis value
    y; m*n array, y-axis value, include y_mean: mean and y_std: std
    Label: str, the label of y value
    Figname: str, the figure name
    '''
    num_line = len(Label)
    plt.style.use('seaborn-whitegrid')

    # color1 = ['rose', 'pale olive', 'cornflower', 'dusty purple', 'gold', 'medium green', 'greyish']
    # color2 = ['red', 'olive', 'blue', 'purple', 'yellow', 'green', 'grey']
    color1 = ['rose', 'medium green', 'cornflower', 'dusty purple', 'gold', 'medium green', 'greyish']
    color2 = ['red', 'green', 'blue', 'purple', 'yellow', 'green', 'grey']

    for i in range(num_line):
        ax = plt.plot(x, y_mean[i, :], label=Label[i])
        plt.setp(ax, color=sns.xkcd_rgb[color1[i]], linewidth=2.0)
        # c = st.norm.interval(alpha=0.8, loc=y_mean[i, :], scale=y_std[i, :])
        # low = c[0]
        # up = c[1]
        low = y_mean[i, :] - y_std[i, :]
        up = y_mean[i, :] + y_std[i, :]
        plt.fill_between(x, low, up, color=color2[i], alpha=0.2, linewidths=0, interpolate=True)
    
    # Gres = 0.68873186e+08 * np.ones(200) * 2
    # Gres = 0.68508142e+08 * np.ones(200) / 2
    Gres = 5.24454384e+8 * np.ones(200) 
    ax = plt.plot(x, Gres, label= 'Greedy Policy')
    plt.setp(ax, color=sns.xkcd_rgb[color1[-1]], linewidth=2.0)


    plt.xlabel('Training episodes', fontsize=12)
    plt.ylabel('Rewards', fontsize=12)
    # plt.title(Figname, fontsize=12)
    plt.legend(fontsize=12)
    # plt.legend(fontsize=12, bbox_to_anchor=(0.7, 0.5))
    plt.savefig(Figname + '.png', dpi=600)
    # plt.savefig(Figname + '.eps')


if __name__ == '__main__':

    data = []
    data1 = []
    data2 = []
    GreedData = []

    ## SIX TURBINES
    with open('MAPO/learning_curves1/Fastgroup_rewards.pkl', 'rb') as f:
        result = pickle.load(f)
        data.append(result[0:200])
    with open('MAPO/learning_curves2/Fastgroup_rewards.pkl', 'rb') as f:
        result = pickle.load(f)
        data.append(result[0:200])
    with open('MAPO/learning_curves3/Fastgroup_rewards.pkl', 'rb') as f:
        result = pickle.load(f)
        data.append(result[0:200])
    with open('MAPO/learning_curves4/Fastgroup_rewards.pkl', 'rb') as f:
        result = pickle.load(f)
        data.append(result[0:200])
    with open('MAPO/learning_curves5/Fastgroup_rewards.pkl', 'rb') as f:
        result = pickle.load(f)
        data.append(result[0:200])
    
     ## SIX TURBINES
    with open('MADDPG/learning_curves1/Fastgroup_rewards.pkl', 'rb') as f:
        result = pickle.load(f)
        data1.append(result[0:200])
    with open('MADDPG/learning_curves2/Fastgroup_rewards.pkl', 'rb') as f:
        result = pickle.load(f)
        data1.append(result[0:200])
    with open('MADDPG/learning_curves3/Fastgroup_rewards.pkl', 'rb') as f:
        result = pickle.load(f)
        data1.append(result[0:200])
    with open('MADDPG/learning_curves4/Fastgroup_rewards.pkl', 'rb') as f:
        result = pickle.load(f)
        data1.append(result[0:200])
    with open('MADDPG/learning_curves5/Fastgroup_rewards.pkl', 'rb') as f:
        result = pickle.load(f)
        data1.append(result[0:200])


    train_step = np.array(range(len(data[0])))
    data = np.array(data)
    data1 = np.array(data1)
    # GreedData = np.array(GreedData)
    # print(data)
    
    data_mean, data_std = mean_and_std(data)
    data1_mean, data1_std = mean_and_std(data1)
    # data2_mean, data2_std = mean_and_std(data2)
    data_mean = np.array([data_mean, data1_mean])
    data_std = np.array([data_std, data1_std])
    # Gdata_mean, Gdata_std = mean_and_std(GreedData)
    # data_mean = np.reshape(data_mean, (1, 200))
    # data_std = np.reshape(data_std, (1, 200))
    label = ['MAPO', "MADDPG"]
    figname = 'Six_Turbines'
    plot_results(train_step, data_mean, data_std, label, figname)