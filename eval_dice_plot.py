import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# LIDC
# experiment_base_folder = '/itet-stor/baumgach/net_scratch/logs/segvae/lidc/'
# experiment_list = ['detUNET',
#                    'final_res128_probunet_1annotator/',
#                    'final_res128_hybrid_7_1_1annotator',
#                    'final_res128_hybrid_7_5_bs12_partdep_1annotator/']
# experiment_names = ['detUNET', 'ProbUNET_1annot', 'SegVAE_1lvls_1annot', 'SegVAE_5lvls_1annot']
# file_list = ['dice_best_dice.npz']*len(experiment_list)

# PROSTATE
# experiment_base_folder = '/itet-stor/baumgach/net_scratch/logs/segvae/uzh_prostate/'
# experiment_list = ['detUNET',
#                    'probUNET_bs12_zdim6_1annotator',
#                    'segvae_7_1_bs12_1annotator',
#                    'segvae_7_5_bs12_1annotator']
# experiment_names = ['detUNET','ProbUNET_1annot', 'SegVAE_1lvls_1annot', 'SegVAE_5lvls_1annot']
# file_list = ['dice_best_dice.npz']*len(experiment_list)

# PROSTATE (TWO LBL)
# experiment_base_folder = '/itet-stor/baumgach/net_scratch/logs/segvae/uzh_prostate_twolbl/'
# experiment_list = ['detUNET_1annotator',
#                    'probUNET_1annotator',
#                    'segvae_7_1_1annot',
#                    'segvae_7_5_1annot_rerun']
# experiment_names = ['detUNET','ProbUNET_1annot', 'SegVAE_1lvls_1annot', 'SegVAE_5lvls_1annot']
# file_list = ['dice_best_dice.npz']*len(experiment_list)

# LIDC RERUN
experiment_base_folder = '/itet-stor/baumgach/net_scratch/logs/phiseg/lidc/'
experiment_list = ['detunet',
                   'probunet_1annot',
                   'segvae_7_1_1annot',
                   'segvae_7_5_1annot']
experiment_names = ['detUNET','ProbUNET_1annot', 'SegVAE_1lvls_1annot', 'SegVAE_5lvls_1annot']
file_list = ['dice_best_dice.npz']*len(experiment_list)


ged_list = []

for folder, exp_name, file in zip(experiment_list, experiment_names, file_list):

    experiment_path = os.path.join(experiment_base_folder, folder, file)

    ged_arr = np.load(experiment_path)['arr_0']

    print(ged_arr.shape)

    ged_list.append(np.mean(ged_arr[:,1:],axis=-1))

ged_tot_arr = np.asarray(ged_list).T

print('significance')
print('REMINDER: are you checking the right methods?')
print(stats.ttest_rel(ged_list[0], ged_list[3]))

print('Results summary')
# means = np.median(ged_tot_arr, axis=0)
means = ged_tot_arr.mean(axis=0)
stds= ged_tot_arr.std(axis=0)

print(means.shape)
print(stds.shape)

for i in range(means.shape[0]):
    print('Exp. name: %s \t %.4f +- %.4f' % (experiment_names[i], means[i], stds[i]))

df = pd.DataFrame(ged_tot_arr, columns=experiment_names)
df = df.melt(var_name='experiments', value_name='vals')

sns.boxplot(x='experiments', y='vals', data=df)
plt.show()