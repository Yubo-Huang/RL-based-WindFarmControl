import matplotlib.pyplot as plt
import pickle

with open('learning_curves/Fast_rewards.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)

# x = range(20)
# plt.plot(x, data)
# plt.show()