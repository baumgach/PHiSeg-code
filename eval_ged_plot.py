import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

experiment_base_folder = '/itet-stor/baumgach/net_scratch/logs/phiseg/lidc/'
experiment_list = ['probunet',
                   'phiseg_7_1',
                   'phiseg_7_5',
                   'probunet_1annot',
                   'phiseg_7_1_1annot',
                   'phiseg_7_5_1annot']
experiment_names = ['probunet','phiseg_7_1', 'phiseg_7_5', 'probunet_1annot', 'phiseg_7_1_1annot', 'phiseg_7_5_1annot']
file_list = ['ged100_best_ged.npz']*len(experiment_list)


ged_list = []

for folder, exp_name, file in zip(experiment_list, experiment_names, file_list):

    experiment_path = os.path.join(experiment_base_folder, folder, file)

    ged_arr = np.load(experiment_path)['arr_0']

    ged_list.append(ged_arr)

ged_tot_arr = np.asarray(ged_list).T

print('significance')
print('REMINDER: are you checking the right methods?')
print(stats.ttest_rel(ged_list[0], ged_list[1]))

print('Results summary')
means = ged_tot_arr.mean(axis=0)
stds= ged_tot_arr.std(axis=0)

for i in range(means.shape[0]):
    print('Exp. name: %s \t %.4f +- %.4f' % (experiment_names[i], means[i], stds[i]))

df = pd.DataFrame(ged_tot_arr, columns=experiment_names)
df = df.melt(var_name='experiments', value_name='vals')

sns.boxplot(x='experiments', y='vals', data=df)
plt.show()