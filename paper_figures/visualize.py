from font_styles import rc_fonts
import matplotlib as mpl
mpl.rcParams.update(rc_fonts[5])
from tqdm import tqdm
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.ticker as mtick
import os
from glob import glob

str2latex = {
            'c11': '$C_{11}$',
            'c12': '$C_{12}$',
            'c44': '$C_{44}$',
            'G': 'G',
            'B': 'B',
            'E_VRH': '$E_{VRH}$',
            'nu': r'$\nu$',
            'Cohesive_energy': '$E_{coh}$'
            }

markers = {
            'c11': '8',
            'c12': '^',
            'c44': 's',
            'G': 'p',
            'B': 'H',
            'E_VRH': 'D',
            'nu': 'h',
            'Cohesive_energy': 'd'
            }

colors = {
            'c11': 'b',
            'c12': 'g',
            'c44': 'r',
            'G': 'c',
            'B': 'm',
            'E_VRH': 'olive',
            'nu': 'purple',
            'Cohesive_energy': 'darkslategray'
            }

def calculate_r2(data, property=None, split=None, type=None):
    assert (split is None and type is not None) or \
           (split is not None and type is None) or \
           (split is None and type is None) , "only one or neither of the 'split' or 'type' args should be provided"
    if property is not None:
        target_key = property
        pred_key = f'{property}_pred'
        if split is not None:
            r2 = r2_score(data[target_key][data['split']==split], data[pred_key][data['split']==split])
        elif type is not None:
            r2 = r2_score(data[target_key][data['type']==type], data[pred_key][data['type']==type])
        else:
            r2 = r2_score(data[target_key], data[pred_key])
    return round(r2, 3)


def calculate_rmse(data, property=None, split=None, type=None):
    assert (split is None and type is not None) or \
           (split is not None and type is None) or \
           (split is None and type is None) , "only one or neither of the 'split' or 'type' args should be provided"
    if property is not None:
        target_key = property
        pred_key = f'{property}_pred'
        if split is not None:
            mse = mean_squared_error(data[target_key][data['split']==split], data[pred_key][data['split']==split])
        elif type is not None:
            mse = mean_squared_error(data[target_key][data['type']==type], data[pred_key][data['type']==type])
        else:
            mse = mean_squared_error(data[target_key], data[pred_key])
        rmse = np.sqrt(mse)
    return round(rmse, 3)


def plot_r2(data_):
    mpl.rcParams.update(rc_fonts[5])
    mpl.rcParams['markers.fillstyle'] = 'full'
    mpl.rcParams['lines.markersize'] = 13
    
    datasets = ['25B', '50B_newseed1234', '75B_newseed72', '100B', '100B_25T', '100B_50T_newseed1234', '100B_75T', '100B_100T']
    # datasets = ['25B', '50B', '75B', '100B', '100B_25T', '100B_50T', '100B_75T', '100B_100T']
    properties = ['c11', 'c12', 'c44', 'G', 'B', 'E_VRH', 'nu', 'Cohesive_energy']

    plt.figure()
    fig, ax = plt.subplots(figsize=(15, 12))

    r2_values = {}
    r2_avg_over_datasets = []

    for dataset in tqdm(datasets):
        filename = f'results/normalized/results_{dataset}.xlsx'
        result = pd.read_excel(filename)
        data = data_
        data['split'] = result['set_type']
        data['split'] = data['split'].replace('training', 'train')
        data['c11_pred'] = result['c11']
        data['c12_pred'] = result['c12']
        data['c44_pred'] = result['c44']
        data['G_pred'] = result['G']
        data['B_pred'] = result['B']
        data['E_VRH_pred'] = result['E_VRH']
        data['nu_pred'] = result['nu']
        data['Cohesive_energy_pred'] = result['Cohesive_energy']
        data.to_csv(f'results/normalized/augmented_results_{dataset}.csv', index=False)

        r2_values[dataset] = {}
        for property in properties:
            data = data[data['type'].isin(['quaternary', 'quinary'])]
            r2 = calculate_r2(data, property=property)
            r2_values[dataset][property] = r2

        r2_avg = np.mean(list(r2_values[dataset].values()))
        r2_avg_over_datasets.append(r2_avg)


    pd.DataFrame(r2_values).to_csv('r2_values.csv')

    r2_values = {property: {key: r2_values[key][property] for key in r2_values} for property in properties}
    x_values = [i for i in range(len(datasets))]

    for property, values in r2_values.items():
        ax.plot(x_values, list(values.values()), label=str2latex[property], linestyle='--', marker=markers[property], color=colors[property])
        # ax.scatter(x_values, list(values.values()), label=str2latex[property], marker=markers[property])

    ax.plot(x_values, r2_avg_over_datasets, label="average", marker='o', color='black', linewidth=4)

    # plt.legend(loc='lower left', frameon=False, bbox_to_anchor=(0.5, 0.6))
    plt.legend(loc='lower right', frameon=False, ncol=5)
    plt.xlabel(f'Percentage of data used for training')
    plt.ylabel(r'$R^2$ score')
    plt.xticks(x_values, [r'$25\% B$', r'$50\% B$', r'$75\% B$', r'$100\% B$', r'$100\% B + 25\% T$', r'$100\% B + 50\% T$', r'$100\% B + 75\% T$', r'$100\% B + 100\% T$']) 

    plt.xticks(rotation=45)
    plt.ylim(0.45,0.9)

    fig.tight_layout(pad=0.30)
    # fig.tight_layout()
    plt.savefig(f'plots/r2_values', facecolor='white', dpi=600)

    return r2_values


def target_vs_preds(data, property=None, color_code_by=None, path_to_save=None):

    if property is None:

        properties = ['c11', 'c12', 'c44', 'G', 'B', 'E_VRH', 'nu', 'Cohesive_energy']
        
        # fig_size = (24, 12) if color_code_by!='type' else (28, 14)
        if color_code_by=='type':
            mpl.rcParams.update(rc_fonts[6])

        fig, axs = plt.subplots(2, 4)
        axs = axs.flatten()

        for i, ax in enumerate(axs):
            property = properties[i]
            target_key = property
            pred_key = f'{property}_pred'
            targets = data[target_key].to_list()
            preds = data[pred_key].to_list()

            low = np.min([targets, preds])
            high = np.max([targets, preds])
            variance = 0.1*(high-low)
            low -= variance
            high += variance

            
            if color_code_by is None:
                
                ax.plot([low,high],[low,high], '--', color='k')
                # data = data[data['type'].isin(['quaternary', 'quinary'])]
                r2 = calculate_r2(data, property=property)
                rmse = calculate_rmse(data, property=property)
                ax.scatter(data[target_key][data['split']=='test'], data[pred_key][data['split']=='test'], color='g', alpha=0.7)
                # ax.text(low, high-variance, s=f'RMSE={rmse}\n\n\n$R^2$={r2}')
                ax.text(low+variance, high-variance, s=f'RMSE={rmse}')
                ax.text(low+variance, high-3*variance, s=f'$R^2$={r2}')
                ax.set_xlim([low,high])
                ax.set_ylim([low,high])

            elif color_code_by=='split':
                ax.plot([low,high],[low,high], '--', color='k')
                r2_train = calculate_r2(data, property=property, split='train')
                rmse_train = calculate_rmse(data, property=property, split='train')

                r2_val = calculate_r2(data, property=property, split='validation')
                rmse_val = calculate_rmse(data, property=property, split='validation')

                r2_test = calculate_r2(data, property=property, split='test')
                rmse_test = calculate_rmse(data, property=property, split='test')

                ax.scatter(data[target_key][data['split']=='train'], data[pred_key][data['split']=='train'], color='r', label=f'train: RMSE={rmse_train}, $R^2$={r2_train}')
                ax.scatter(data[target_key][data['split']=='validation'], data[pred_key][data['split']=='validation'], color='b', label=f'validation: RMSE={rmse_val}, $R^2$={r2_val}')
                ax.scatter(data[target_key][data['split']=='test'], data[pred_key][data['split']=='test'], color='g', label=f'test: RMSE={rmse_test}, $R^2$={r2_test}')


            elif color_code_by=='type':

                # low += variance
                # high -= variance
                ax.plot([low,high],[low,high], '--', color='k')
                
                r2_bin = calculate_r2(data, property=property, type='binary')
                rmse_bin = calculate_rmse(data, property=property, type='binary')

                r2_ter = calculate_r2(data, property=property, type='ternary')
                rmse_ter = calculate_rmse(data, property=property, type='ternary')

                r2_qua = calculate_r2(data, property=property, type='quaternary')
                rmse_qua = calculate_rmse(data, property=property, type='quaternary')

                r2_qui = calculate_r2(data, property=property, type='quinary')
                rmse_qui = calculate_rmse(data, property=property, type='quinary')

                ax.scatter(data[target_key][data['type']=='binary'], data[pred_key][data['type']=='binary'], color='k', label=f'$R_{{Binary}}^2$={r2_bin}', alpha=0.7)
                ax.scatter(data[target_key][data['type']=='ternary'], data[pred_key][data['type']=='ternary'], color='b', label=f'$R_{{Ternary}}^2$={r2_ter}', alpha=0.7)
                ax.scatter(data[target_key][data['type']=='quaternary'], data[pred_key][data['type']=='quaternary'], color='g', label=f'$R_{{Quaternary}}^2$={r2_qua}', alpha=0.7)
                ax.scatter(data[target_key][data['type']=='quinary'], data[pred_key][data['type']=='quinary'], color='r', label=f'$R_{{Quinary}}^2$={r2_qui}', alpha=0.7)
                ax.set_xlim([low,high])
                ax.set_ylim([low,high])
            # if i!=7:
            #     ax.legend(loc='upper left', frameon=False)
            # else:
            #     ax.legend(loc='lower right', frameon=False)

            ax.legend(loc='upper left', frameon=False, handletextpad=0.0, bbox_to_anchor=(0, 1))

            
            ax.set_xlabel(f'Actual {str2latex[target_key]}')
            ax.set_ylabel(f'Predicted {str2latex[target_key]}')

        # plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
        # plt.subplots_adjust(hspace=0.3)
        fig.tight_layout(pad=0.8)
        if path_to_save is None:
            plt.savefig(f'plots/collective.png', facecolor='white', dpi=600)
        else:
            plt.savefig(f'{path_to_save}', facecolor='white', dpi=600)

    else:

        target_key = property
        pred_key = f'{property}_pred'
        targets = data[target_key].to_list()
        preds = data[pred_key].to_list()

        low = np.min([targets, preds])
        high = np.max([targets, preds])
        variance = 0.1*(high-low)
        low -= variance
        high += variance

        plt.figure()
        fig, ax = plt.subplots()

        plt.plot([low,high],[low,high], '--', color='k')

        if color_code_by is None:
            r2 = calculate_r2(data, property=property, split='test')
            rmse = calculate_rmse(data, property=property, split='test')
            plt.scatter(data[target_key][data['split']=='test'], data[pred_key][data['split']=='test'], color='g')
            plt.text(low, high-variance, s=f'RMSE={rmse}, $R^2$={r2}')
        
        elif color_code_by=='split':

            r2_train = calculate_r2(data, property=property, split='train')
            rmse_train = calculate_rmse(data, property=property, split='train')

            r2_val = calculate_r2(data, property=property, split='validation')
            rmse_val = calculate_rmse(data, property=property, split='validation')

            r2_test = calculate_r2(data, property=property, split='test')
            rmse_test = calculate_rmse(data, property=property, split='test')

            plt.scatter(data[target_key][data['split']=='train'], data[pred_key][data['split']=='train'], color='r', label=f'train: RMSE={rmse_train}, $R^2$={r2_train}')
            plt.scatter(data[target_key][data['split']=='validation'], data[pred_key][data['split']=='validation'], color='b', label=f'validation: RMSE={rmse_val}, $R^2$={r2_val}')
            plt.scatter(data[target_key][data['split']=='test'], data[pred_key][data['split']=='test'], color='g', label=f'test: RMSE={rmse_test}, $R^2$={r2_test}')

        elif color_code_by=='type':

            r2_bin = calculate_r2(data, property=property, type='binary')
            rmse_bin = calculate_rmse(data, property=property, type='binary')

            r2_ter = calculate_r2(data, property=property, type='ternary')
            rmse_ter = calculate_rmse(data, property=property, type='ternary')

            r2_qua = calculate_r2(data, property=property, type='quaternary')
            rmse_qua = calculate_rmse(data, property=property, type='quaternary')

            r2_qui = calculate_r2(data, property=property, type='quinary')
            rmse_qui = calculate_rmse(data, property=property, type='quinary')

            plt.scatter(data[target_key][data['type']=='binary'], data[pred_key][data['type']=='binary'], color='k', label=f'binary: RMSE={rmse_bin}, $R^2$={r2_bin}')
            plt.scatter(data[target_key][data['type']=='ternary'], data[pred_key][data['type']=='ternary'], color='b', label=f'ternary: RMSE={rmse_ter}, $R^2$={r2_ter}')
            plt.scatter(data[target_key][data['type']=='quaternary'], data[pred_key][data['type']=='quaternary'], color='g', label=f'quaternary: RMSE={rmse_qua}, $R^2$={r2_qua}')
            plt.scatter(data[target_key][data['type']=='quinary'], data[pred_key][data['type']=='quinary'], color='r', label=f'quinary: RMSE={rmse_qui}, $R^2$={r2_qui}')

        plt.legend(loc='upper left', frameon=False)
        plt.xlabel(f'Actual {target_key}')
        plt.ylabel(f'Predicted {target_key}')

        # fig.tight_layout(pad=0.25)

        plt.savefig(f'plots/{target_key}', facecolor='white', dpi=600)


def box_plot(data, labeled_by='split'):

    properties = ['c11', 'c12', 'c44', 'G', 'B', 'E_VRH', 'nu', 'Cohesive_energy']
    fig, axs = plt.subplots(4, 2, figsize=(12, 24))
    axs = axs.flatten()

    if labeled_by=='split':

        for i, ax in enumerate(axs):

            property = properties[i]

            train = data[property][data['split']=='train'].to_numpy()
            validation = data[property][data['split']=='validation'].to_numpy()
            test = data[property][data['split']=='test'].to_numpy()

            ax.boxplot([train, validation, test])

            ax.set_xticks([1, 2, 3], ['train', 'validation', 'test'])

            ax.set_ylabel(f'{property}')
    
    elif labeled_by=='type':

        for i, ax in enumerate(axs):

            property = properties[i]

            binary = data[property][data['type']=='binary'].to_numpy()
            ternary = data[property][data['type']=='ternary'].to_numpy()
            quaternary = data[property][data['type']=='quaternary'].to_numpy()
            quinary = data[property][data['type']=='quinary'].to_numpy()

            ax.boxplot([binary, ternary, quaternary, quinary])

            ax.set_xticks([1, 2, 3, 4], ['binary', 'ternary', 'quaternary', 'quinary'])

            ax.set_ylabel(f'{property}')

    fig.tight_layout(pad=0.15)
    plt.savefig(f'plots/box.png', facecolor='white', dpi=600)


def hist_plot(data, labeled_by='split'):

    mpl.rcParams['legend.fontsize'] = 20
    properties = ['c11', 'c12', 'c44', 'G', 'B', 'E_VRH', 'nu', 'Cohesive_energy']
    fig, axs = plt.subplots(2, 4, figsize=(28, 12))
    axs = axs.flatten()

    if labeled_by=='split':

        for i, ax in enumerate(axs):

            property = properties[i]

            train = data[property][data['split']=='train'].to_numpy()
            validation = data[property][data['split']=='validation'].to_numpy()
            test = data[property][data['split']=='test'].to_numpy()

            ax.hist(train, bins=20, histtype='step', color='r', label='Training Data')
            ax.hist(validation, bins=20, histtype='step', color='b', label='Validation Data')
            ax.hist(test, bins=20, histtype='step', color='g', label='Test Data')

            ax.set_ylabel(f'{property}')

            if i==1:
                ax.legend(loc='upper right', frameon=False)
    
    
    elif labeled_by=='type':

        for i, ax in enumerate(axs):

            property = properties[i]

            binary = data[property][data['type']=='binary'].to_numpy()
            ternary = data[property][data['type']=='ternary'].to_numpy()
            quaternary = data[property][data['type']=='quaternary'].to_numpy()
            quinary = data[property][data['type']=='quinary'].to_numpy()

            # Normalize counts to percentages

            total_counts = [len(binary), len(ternary), len(quaternary), len(quinary)]
            weights_binary = np.ones_like(binary) / sum(total_counts) * 100
            weights_ternary = np.ones_like(ternary) / sum(total_counts) * 100
            weights_quaternary = np.ones_like(quaternary) / sum(total_counts) * 100
            weights_quinary = np.ones_like(quinary) / sum(total_counts) * 100
            weights = [weights_binary, weights_ternary, weights_quaternary, weights_quinary]

            counts1, bins1, patches1 = ax.hist(binary, bins=10, weights=weights[0], edgecolor='k', color='k', label='Binary', alpha=0.5)
            counts1, bins1, patches1 = ax.hist(ternary, bins=bins1, weights=weights[1], edgecolor='k', color='b', label='Ternary', alpha=0.5)
            ax1 = ax.twinx()
            counts1, bins1, patches1 = ax1.hist(quaternary, bins=bins1, weights=weights[2], edgecolor='k', color='g', label='Quternary', alpha=0.5)
            counts1, bins1, patches1 = ax1.hist(quinary, bins=bins1, weights=weights[3], edgecolor='k', color='r', label='Quinary', alpha=0.5)

            ax.set_xticks(np.linspace(min(bins1), max(bins1), 4))
            ax.set_xlabel(f'{str2latex[property]}')

            if i==0:
                ax.set_ylabel(f'Bin. & Tern.')
                ax1.set_ylabel(f'Quat. & Quin.')
                ax.legend(loc='lower left', frameon=False, bbox_to_anchor=(0.48,0.7))
                ax1.legend(loc='lower left', frameon=False, bbox_to_anchor=(0.48,0.55))

    fig.tight_layout(pad=1)
    plt.savefig(f'plots/hist.png', facecolor='white', dpi=600)




def plot_histogram():
    data = pd.read_csv('data.csv')
    hist_plot(data, labeled_by='type')

def plot_r2_values():
    data = pd.read_csv('data.csv')
    r2_values = plot_r2(data)

def plot_qq_100B():
    data = pd.read_csv('results/augmented_results_100B.csv')
    data = data[data['type'].isin(['quaternary', 'quinary'])]
    target_vs_preds(data, color_code_by=None)

def plot_qq_100B_100T():
    data = pd.read_csv('results/augmented_results_100B_100T.csv')
    data = data[data['type'].isin(['quaternary', 'quinary'])]
    target_vs_preds(data, color_code_by=None)


def plot_per_type_100B_100T():
    data = pd.read_csv('results/augmented_results_100B_100T.csv')
    target_vs_preds(data, color_code_by='type')

def plot_per_type_100B():
    data = pd.read_csv('results/augmented_results_100B.csv')
    target_vs_preds(data, color_code_by='type')

def plot_per_type_100B_2():
    data = pd.read_csv('results/100%Binaries(No normalization)/augmented_results_100B.csv')
    target_vs_preds(data, color_code_by='type')

def plot_qq_100B_2():
    data = pd.read_csv('results/100%Binaries(No normalization)/augmented_results_100B.csv')
    data = data[data['type'].isin(['quaternary', 'quinary'])]
    target_vs_preds(data, color_code_by=None)

def plot_per_type_100B_100T_2():
    data = pd.read_csv('results/100%B100%T(without normalization)/augmented_results_100B_100T.csv')
    target_vs_preds(data, color_code_by='type')

def plot_qq_100B_100T_2():
    data = pd.read_csv('results/100%B100%T(without normalization)/augmented_results_100B_100T.csv')
    data = data[data['type'].isin(['quaternary', 'quinary'])]
    target_vs_preds(data, color_code_by=None)



## Following codes are for plotting the results of the models trained on normalized data
# plot_qq_100B()
# plot_qq_100B_100T()
# plot_per_type_100B()
# plot_per_type_100B_100T()


"""
Uncomment each of the following commands to reproduce the figures stored in "plots" folder
""" 

# plot_histogram()
# plot_r2_values()
# plot_qq_100B_2()
# plot_per_type_100B_2()
# plot_qq_100B_100T_2()
# plot_per_type_100B_100T_2()
# print()

