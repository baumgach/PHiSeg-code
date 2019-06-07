import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# PROSTATE NCC 100 samps
# experiment_base_folder = '/itet-stor/baumgach/net_scratch/logs/phiseg/uzh_prostate/'
# experiment_list = ['probUNET_bs12_zdim6',
#                    'segvae_7_1_bs12',
#                    'segvae_7_5_bs12',
#                    'probUNET_bs12_zdim6_1annotator',
#                    'segvae_7_1_bs12_1annotator',
#                    'segvae_7_5_bs12_1annotator']
# experiment_names = ['ProbUNET','SegVAE_1lvls', 'SegVAE_5lvls','ProbUNET_1annot', 'SegVAE_1lvls_1annot', 'SegVAE_5lvls_1annot']
# file_list = ['ncc100_best_loss.npz']*len(experiment_list)

# # LIDC NCC 100 samps
# experiment_base_folder = '/itet-stor/baumgach/net_scratch/logs/phiseg/lidc/'
# experiment_list = ['final_res128_probunet_bn_bs12_zdim6',
#                    'final_res128_hybrid_7_1_rerun',
#                    'final_res128_hybrid_7_5_bs12_partdep',
#                    'final_res128_probunet_1annotator/',
#                    'final_res128_hybrid_7_1_1annotator',
#                    'final_res128_hybrid_7_5_bs12_partdep_1annotator/']
# experiment_names = ['ProbUNET','SegVAE_1lvls', 'SegVAE_5lvls','ProbUNET_1annot', 'SegVAE_1lvls_1annot', 'SegVAE_5lvls_1annot']
# file_list = ['ncc100_best_loss.npz']*len(experiment_list)

# # PROSTATE NCC 100 samps
# experiment_base_folder = '/itet-stor/baumgach/net_scratch/logs/phiseg/uzh_prostate_twolbl/'
# experiment_list = ['probUNET',
#                    'segvae_7_1',
#                    'segvae_7_5',
#                    'probUNET_1annotator_rerun_wd',
#                    'segvae_7_1_1annot',
#                    'segvae_7_5_1annot']
# experiment_names = ['ProbUNET','SegVAE_1lvls', 'SegVAE_5lvls','ProbUNET_1annot', 'SegVAE_1lvls_1annot', 'SegVAE_5lvls_1annot']
# file_list = ['ncc100_best_loss.npz']*len(experiment_list)

# LIDC RERUN
# experiment_base_folder = '/itet-stor/baumgach/net_scratch/logs/phiseg/lidc/'
# experiment_list = ['probunet',
#                    'segvae_7_1',
#                    'segvae_7_5',
#                    'probunet_1annot',
#                    'segvae_7_1_1annot',
#                    'segvae_7_5_1annot']
# experiment_names = ['ProbUNET','SegVAE_1lvls', 'SegVAE_5lvls','ProbUNET_1annot', 'SegVAE_1lvls_1annot', 'SegVAE_5lvls_1annot']
# file_list = ['ncc100_best_loss.npz']*len(experiment_list)
#

# PROSTATE NCC 100 samps (AFTER MICCAI)
experiment_base_folder = '/itet-stor/baumgach/net_scratch/logs/phiseg/uzh_prostate_afterpaper/'
experiment_list = ['probUNET',
                   'segvae_7_5',
                   'probUNET_1annotator_2',
                   'segvae_7_5_1annot']
experiment_names = ['ProbUNET', 'SegVAE_5lvls','ProbUNET_1annot', 'SegVAE_5lvls_1annot']
file_list = ['ncc100_best_ged.npz']*len(experiment_list)

ged_list = []

for folder, exp_name, file in zip(experiment_list, experiment_names, file_list):

    experiment_path = os.path.join(experiment_base_folder, folder, file)

    ged_arr = np.squeeze(np.load(experiment_path)['arr_0'])

    ged_list.append(ged_arr)

ged_tot_arr = np.asarray(ged_list).T

print('significance')
print('REMINDER: are you checking the right methods?')
print(stats.ttest_rel(ged_list[2], ged_list[3]))

print('Results summary')
means = ged_tot_arr.mean(axis=0)
stds= ged_tot_arr.std(axis=0)

print(ged_tot_arr.shape)

for i in range(means.shape[0]):
    print('Exp. name: %s \t %.4f +- %.4f' % (experiment_names[i], means[i], stds[i]))

df = pd.DataFrame(ged_tot_arr, columns=experiment_names)
df = df.melt(var_name='experiments', value_name='vals')

sns.boxplot(x='experiments', y='vals', data=df)
plt.show()