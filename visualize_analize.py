import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, silhouette_score

parser = ArgumentParser()
parser.add_argument('--raw-data', type=str, default='cyclo70',
                    help='path to folder containing the .dat files')
parser.add_argument('--reference-value', type=str, default='reference_values/dlpno-ccsdt-34.dat',
                    help='path to the reference file')
parser.add_argument('--old-reference-value', type=str, default='reference_values/dlpno-ccsdt-23.dat',
                    help='path to the old reference file')
parser.add_argument('--full-data', type=str, default='cyclo70/full_data.csv',
                    help='path to the full data file')
parser.add_argument('--class-data', type=str, default='complementary_files/reaction_classes_cyclo70.csv',
                    help='path to the classes file')
parser.add_argument('--full-data-bh9', type=str, default='bh9/full_data.csv',
                    help='path to the full data BH9 file')
parser.add_argument('--class-data-bh9', type=str, default='complementary_files/reaction_classes_bh9.csv',
                    help='path to the classes BH9 file')
parser.add_argument('--rung-data', type=str, default='complementary_files/features_csv.csv',
                    help='path to the classes file')
parser.add_argument('--std', action='store_true', default=False, help='Initial preparation of the data')
parser.add_argument('--error', action='store_true', default=False, help='Compute MAE')
parser.add_argument('--pca', action='store_true', default=False, help='Principal Component Analysis')
parser.add_argument('--corr-reference', action='store_true', default=False, help='Correlation between DZ --> TZ and TZ --> QZ')
parser.add_argument('--std-plot', action='store_true', default=False, help='Plot standard deviation')
parser.add_argument('--mae-plot', action='store_true', default=False, help='Plot standard deviation')
parser.add_argument('--std-plot-jctc', action='store_true', default=False, help='Plot standard deviation with the data of the JCTC paper')


functionals = {'wB2PLYP': '$\omega$B2PLYP', 'wB2PLYP18':'$\omega$B2PLYP18', 'wB2GP-PLYP':'$\omega$B2GP-PLYP',
               'r2SCAN0-2': 'r$^2$SCAN0-2', 'kPr2SCAN50': '$\kappa$Pr$^2$SCAN50', 'Pr2SCAN50': 'Pr$^2$SCAN50',
               'Pr2SCAN69': 'Pr$^2$SCAN69', 'wB88PP86': '$\omega$B88PP86', 'wPBEPP86': '$\omega$PBEPP86', 'wB97X-2': '$\omega$B97X-2',
               'wPr2SCAN50': '$\omega$Pr$^2$SCAN50', 'wB97M': '$\omega$B97M', 'wB97M-D3BJ': '$\omega$B97M-D3BJ', 'wB97M-D4': '$\omega$B97M-D4',
               'wB97M-D4rev': '$\omega$B97M-D4rev', 'wB97M-V': '$\omega$B97M-V', 'wB97X': '$\omega$B97X', 'wB97X-D3BJ': '$\omega$B97X-D3BJ',
               'wB97X-D3': '$\omega$B97X-D3', 'wB97X-D4': '$\omega$B97X-D4', 'wB97X-D4rev': '$\omega$B97X-D4rev', 'wr2SCAN' : '$\omega$r$^2$SCAN',
               'wB97X-V': '$\omega$B97X-V', 'r2SCAN0' : 'r$^2$SCAN0', 'r2SCAN50' : 'r$^2$SCAN50',  'r2SCANh' : 'r$^2$SCANh',  'r2SCAN' : 'r$^2$SCAN',
               'wB97': '$\omega$B97',}


def combine_and_compute_std(folder_path):
    all_dfs = []
    # Loop through all .csv files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(folder_path, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file_path, index_col=0)
            all_dfs.append(df)

    merged_df = all_dfs[0]
    for df in all_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=['Name'])

    dft_forward_columns = merged_df.filter(like='DFT-forward')
    dft_reverse_columns = merged_df.filter(like='DFT-reverse')
    dft_reaction_columns = merged_df.filter(like='DFT-reaction')

    merged_df['Range_DFT_forward'] = dft_forward_columns.apply(lambda row: row.max() - row.min(), axis=1)
    merged_df['Std_DFT_forward'] = dft_forward_columns.std(axis=1)

    merged_df['Range_DFT_reverse'] = dft_reverse_columns.apply(lambda row: row.max() - row.min(), axis=1)
    merged_df['Std_DFT_reverse'] = dft_reverse_columns.std(axis=1)

    merged_df['Range_DFT_reaction'] = dft_reaction_columns.apply(lambda row: row.max() - row.min(), axis=1)
    merged_df['Std_DFT_reaction'] = dft_reaction_columns.std(axis=1)

    for column in ['Range_DFT_forward', 'Std_DFT_forward', 'Range_DFT_reverse',
                   'Std_DFT_reverse', 'Range_DFT_reaction', 'Std_DFT_reaction']:
        merged_df = merged_df.loc[abs(merged_df[column]) <= 1000]

    merged_df.to_csv(f'{folder_path}/full_data.csv')

def read_dat_files_to_dataframes(folder_path):
    # Loop through all .dat files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.dat'):
            dat_file_path = os.path.join(folder_path, filename)

            functional = filename[:-4]

            # Read the .dat file into a DataFrame using regex to handle varying spaces
            try:
                df = pd.read_csv(dat_file_path, sep='\s+', skiprows=1, engine='python', 
                                 names=['Name', f'{functional}-DFT-forward', 
                                        f'{functional}-DFT-reverse', 
                                        f'{functional}-DFT-reaction'])
                for column in df.columns:
                    df[column] = df[column].apply(lambda x: remove_vert_line(x))

                df.to_csv(f"{folder_path}/{filename[:-4]}.csv")

            except pd.errors.ParserError:
                print(f"Warning: Skipping file '{filename}' due to a parsing error.")

def remove_vert_line(entry):
    if str(entry)[-1] == '|':
        return entry[:-1]
    else:
        return entry

def calculate_mae(reference_file, full_data, exx_data):

    df_ref = pd.read_csv(reference_file, sep="\s+")
    df_fx = pd.read_csv(full_data)
    df_exx = pd.read_csv(exx_data, sep=';', index_col=0)

    df_fx_rxn = df_fx.loc[df_fx['Name'].isin(df_ref['R_ID'])]
    df_fx_rxn.reset_index(inplace=True)
    df_fx_rxn = df_fx_rxn.drop(columns=['index', 'Unnamed: 0'])

    df_fx_rxn = pd.concat([df_fx_rxn, df_ref], axis=1)

    for column in df_fx_rxn.columns[1:-10]:

        if 'forward' in column:
            reference_column = 'FORWARD'
        elif 'reverse' in column:
            reference_column = 'REVERSE'
        else:
            reference_column = 'REACTION'
        df_fx_rxn[f"{column}-DELTA"] = abs(df_fx_rxn[column] - df_fx_rxn[reference_column])
        df_fx_rxn[f"{column}-DELT_2"] = (df_fx_rxn[column] - df_fx_rxn[reference_column])**2
        df_fx_rxn[f"{column}-SIGNED"] = df_fx_rxn[column] - df_fx_rxn[reference_column]

    delta_columns = [column for column in df_fx_rxn.columns if 'DELTA' in column]
    signed_columns = [column for column in df_fx_rxn.columns if 'SIGNED' in column]
    squared_columns = [column for column in df_fx_rxn.columns if 'DELT_2' in column]

    dict_fx = {}
    df_fx_rxn.to_csv('aaaaa.csv')
    columns_list = [delta_columns, signed_columns, squared_columns]

    for i, columns in enumerate(columns_list):

        for column in columns:
            if 'forward' in column:
                if i == 0:
                    functional = column[:-18]
                    column_reverse = f'{functional}-DFT-reverse-DELTA'
                elif i == 1:
                    functional = column[:-19]
                    column_reverse = f'{functional}-DFT-reverse-SIGNED'
                elif i == 2:
                    functional = column[:-19]
                    column_reverse = f'{functional}-DFT-reverse-DELT_2'

                mean = df_fx_rxn[[column, column_reverse]].to_numpy().mean()
                
                if i == 1:
                    max = df_fx_rxn[[column, column_reverse]].to_numpy().max()
                    min = df_fx_rxn[[column, column_reverse]].to_numpy().min()
                    dict_fx[functional] = dict_fx.get(functional, []) + [max]
                    dict_fx[functional] = dict_fx.get(functional, []) + [min]
                if i == 2:
                    mean = np.sqrt(mean)

                dict_fx[functional] = dict_fx.get(functional, []) + [mean]

                if i == 0:
                    std = df_fx_rxn[[column, column_reverse]].to_numpy().std()
                    dict_fx[functional] = dict_fx.get(functional, []) + [std]

            elif 'reaction' in column:
                if i == 0:
                    functional = column[:-19]
                else:
                    functional = column[:-20]
                
                if i == 1:
                    max = df_fx_rxn[column].max()
                    min = df_fx_rxn[column].min()
                    dict_fx[functional] = dict_fx.get(functional, []) + [max]
                    dict_fx[functional] = dict_fx.get(functional, []) + [min]

                mean = df_fx_rxn[column].mean()

                if i == 2:
                    mean = np.sqrt(mean)
                dict_fx[functional] = dict_fx.get(functional, []) + [mean]

                if i == 0:
                    std = df_fx_rxn[[column]].to_numpy().std()
                    dict_fx[functional] = dict_fx.get(functional, []) + [std]

    df_mae = pd.DataFrame.from_dict(dict_fx, orient='index', columns=['MAE barrier', 'STD barrier', 'MAE reaction', 'STD reaction', 'Max BH', 'Min BH', 'MSE barrier', 'Max RE', 'Min RE', 'MSE reaction', 'RMSD barrier', 'RMSD reaction'])
    df_mae.index.names = ['Methods']
    df_mae.reset_index(inplace=True)
    
    df_mae["Rung"] = df_mae['Methods'].apply(lambda x: df_exx.loc[x].rung)
    df_mae["rate_barrier"] = df_mae["MAE barrier"]/df_mae["RMSD barrier"]
    df_mae["rate_reaction"] = df_mae["MAE reaction"]/df_mae["RMSD reaction"]

    rung_order = {"DH": 0, "RS-hybrid": 1, "hybrid": 2, "mGGA": 3, "GGA": 4, "LSDA": 5, "other": 6}
    df_mae["Rung"] = pd.Categorical(df_mae["Rung"], categories=rung_order.keys(), ordered=True)
    df_mae = df_mae.round(2)

    dict_df = {'BH': "MAE barrier", 'RE': "MAE reaction"}

    for key in dict_df.keys():
        df_mae = df_mae.sort_values(["Rung", dict_df[key]])
        df_mae.to_csv(f"mae_ordered_{key}.csv")

        df_mae['Methods'] = df_mae['Methods'].replace(functionals)
        plt.figure(figsize=(16, 6))
        sns.set_theme(style="white")
        plot = sns.barplot(x='Methods', y=f'{dict_df[key]}', hue='Rung', data=df_mae, legend=True, palette="dark")
        plot.set_title('')

        if key == 'BH':
            plt.ylim((0, 17))
            y_label = '$\Delta$E$^{\ddag}$'
        elif key == 'RE':
            plt.ylim((0, 31))
            y_label = '$\Delta_r$E'

        plot.legend(title=None)
        plt.axhline(y=1.40, color='black', linestyle='dashed')
        plt.xticks(rotation=90, fontsize=10)
        plt.ylabel(f'MAD {y_label}' + ' (kcal mol$^{-1}$)')
        plt.xlabel('')

        plt.tight_layout()
        plt.savefig(f'mae_95_approaches_{key}.pdf', dpi=300)
    
    plt.clf()

    subset_functionals = ['PBE-QIDH', 'kPr2SCAN50', 'revDOD-PBEP86-D4_2021', 'wB97X-2', 'wB2PLYP18', 'wB97M-V', 'wB97M-D4rev', 'wB97M-D3BJ', 'r2SCAN50', 'M062X', 'r2SCAN0']
    metric = 'SIGNED'

    boxplot_data = []

    for col in df_fx_rxn.columns:
        if metric in col and not 'reaction' in col:
            functional = col[:-19]
            for func in subset_functionals:
                if func == functional:
                    for val in df_fx_rxn[col]:
                        boxplot_data.append({'Methods': func, 'Error': val, 'Metric': metric})

    df_boxplot = pd.DataFrame(boxplot_data)
    df_boxplot["Rung"] = df_boxplot['Methods'].apply(lambda x: df_exx.loc[x].rung)
    #import pdb; pdb.set_trace()
    df_boxplot["Rung"] = pd.Categorical(df_boxplot["Rung"], categories=rung_order.keys(), ordered=True)
    df_boxplot['Methods'] = df_boxplot['Methods'].replace(functionals)
    df_boxplot = df_boxplot.sort_values(["Rung"])
    

    plt.figure(figsize=(10, 6))
    plt.axhline(0.0, color='black', linewidth=1.2, linestyle='--')
    sns.set_style('ticks')
    sns.violinplot(data=df_boxplot, x='Methods', y='Error',
                   hue='Rung', hue_order=['DH', 'RS-hybrid', 'hybrid'], palette="pastel",   
                   cut=2, linewidth=0, density_norm='width', inner=None, width=0.9, alpha=0.4)

    sns.boxplot(data=df_boxplot, x='Methods', y='Error', 
                fliersize=4, flierprops=dict(marker='o', markerfacecolor='black', alpha=1), showmeans=True,
                meanprops={"marker": "+", "markeredgecolor": "black", "markersize": "12"},
                width=0.4, hue='Rung', linewidth=2, hue_order=['DH', 'RS-hybrid', 'hybrid'])
    plt.title(f'')
    plt.ylabel("$\Delta$$\Delta$E $\Delta$E$^{\ddag}$ (kcal mol$^{-1}$)")
    plt.xlabel("")
    plt.xticks(rotation=90)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="", loc='best')    
    plt.tight_layout()

    plt.savefig('boxplot_bh.pdf', dpi=450)
    plt.savefig('boxplot_bh.png', dpi=450)


def build_boxplot_data(df_fx_rxn, metric, subset_functionals, reaction_mode):
    data = []
    for col in df_fx_rxn.columns:
        if metric in col:
            if reaction_mode == 'BH' and 'DFT-reaction' not in col:
                target = True
            elif reaction_mode == 'RE' and 'DFT-reaction' in col:
                target = True
            else:
                target = False

            if target:
                for func in subset_functionals:
                    if func in col:
                        for val in df_fx_rxn[col]:
                            data.append({'Methods': func, 'Error': val, 'Metric': metric, 'Type': reaction_mode})
    return pd.DataFrame(data)

def pca_plots(reference_values, full_data, exx_data, error_data):

    df_ref = pd.read_csv(reference_values, sep="\s+")
    df_fx = pd.read_csv(full_data)
    df_fx_rxn = df_fx.loc[df_fx['Name'].isin(df_ref['R_ID'])]
    df_fx_rxn.reset_index(inplace=True)
    df_fx_rxn = df_fx_rxn.drop(columns=['index', 'Unnamed: 0'])
    df_fx_rxn = pd.concat([df_fx_rxn, df_ref], axis=1)

    for column in df_fx_rxn.columns[1:-10]:
        if 'forward' in column:
            reference_column = 'FORWARD'
        elif 'reverse' in column:
            reference_column = 'REVERSE'
        else:
            reference_column = 'REACTION'
        #df_fx_rxn[f"{column}-DELTA"] = abs(df_fx_rxn[column] - df_fx_rxn[reference_column])
        df_fx_rxn[f"{column}-DELTA"] = df_fx_rxn[column] - df_fx_rxn[reference_column]

    columns = [column for column in df_fx_rxn.columns if 'DELTA' in column]
    
    df_forward = create_df_pca(df_fx_rxn, columns, 'forward')
    df_reverse = create_df_pca(df_fx_rxn, columns, 'reverse')
    df_barrier = pd.concat([df_forward, df_reverse], axis=1)
    df_reaction = create_df_pca(df_fx_rxn, columns, 'reaction')
    perform_pca(df_barrier, 'barrier', exx_data, error_data)
    perform_pca(df_reaction, 'reaction', exx_data, error_data)

def perform_pca(df, keyword, exx_data, error_data):

    std_scaler = StandardScaler()
    scaled_df = std_scaler.fit_transform(df)

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_df)
    pca_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
    
    pca_df['method'] = df.index
    #pca_df = pca_df.loc[ ~ pca_df['method'].isin(['MP2', 'HF', 'HFS'])]

    #pca_df.to_csv(f'cluster_pca_{keyword}_all.csv')

    # Remove some outliers
    #pca_df.drop(pca_df.loc[pca_df['PC1'] == pca_df['PC1'].max()].index, inplace=True)  # HFS is always the outliers in PC1

    #for _ in range(1):
    #    pca_df.drop(pca_df.loc[pca_df['PC1'] == pca_df['PC1'].max()].index, inplace=True) # HFS
 
    #if keyword == 'reaction':
    #    for _ in range(3):
    #        pca_df.drop(pca_df.loc[pca_df['PC2'] == pca_df['PC2'].min()].index, inplace=True) # VWN, VWN3, PWLDA
    

    with open('silhoutte_score.txt', 'a') as f:
        f.write(f'{keyword}\n')
        f.write(f'Explained variance: {pca.explained_variance_ratio_}\n')
    clusters = clustering(pca_df)
    pca_df['labels'] = clusters

    df_exx = pd.read_csv(exx_data, sep=';', index_col=0)
    df_error = pd.read_csv(error_data, index_col=1)
    pca_df["Rung"] = pca_df['method'].apply(lambda x: df_exx.loc[x].rung)
    pca_df["EXX"] = pca_df['method'].apply(lambda x: df_exx.loc[x].EXX)
    pca_df["PT2"] = pca_df['method'].apply(lambda x: df_exx.loc[x].EPT2)
    pca_df["MSD"] = pca_df['method'].apply(lambda x: df_error.loc[x][f"MSE {keyword}"])
    #pca_df = pca_df[pca_df["Rung"] != "other"]
    pca_df.to_csv(f'cluster_pca_{keyword}.csv')

    cluster_labels = pca_df['labels'].unique()  # Get unique cluster labels
    cluster_palette = {label: color for label, color in zip(cluster_labels, 
    ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf'])}
    rung_order = ['DH', 'RS-hyb', 'hybrid', 'mGGA', 'GGA', 'LSDA']

    # Create a FacetGrid for separate panels
    g = sns.FacetGrid(pca_df, col="Rung", col_order=rung_order, col_wrap=3, height=4, sharex=True, sharey=True)

    # Apply scatterplot to each facet
    g.map_dataframe(scatterplot, pca_df=pca_df, cluster_palette=cluster_palette)

    # Set labels and titles
    g.set_axis_labels("", "")
    g.set_titles(col_template="{col_name}")

    # Remove inner ticks and spines, but keep the outer ones
    for i, ax in enumerate(g.axes.flat):
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)

    # Add "PC1" and "PC2" as external labels
    plt.figtext(0.5, -0.02, "PC1", ha="center", fontsize=14)  # Centered below plots
    plt.figtext(-0.02, 0.5, "PC2", va="center", rotation=90, fontsize=14)  # Left side

    plt.tight_layout()
    plt.savefig(f'pca_facet_{keyword}.pdf', dpi=300)

def scatterplot(data, pca_df, cluster_palette, **kwargs):
    ax = plt.gca()
    
    # Plot ALL points in gray in the background
    sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'], color='gray', alpha=0.2, ax=ax, marker='o')

    # Highlight only the points belonging to the current "Rung" panel
    sns.scatterplot(x=data['PC1'], y=data['PC2'], hue=data['labels'], 
                    palette=cluster_palette,
                    style=data['Rung'],
                    edgecolor="black", linewidth=1.2,
                    legend=False, ax=ax)

def clustering(reduced_data):

    range_clusters = [3, 4, 5, 6, 7, 8]
    silhouette_avg = []

    sum_of_squared_distances = []
   
    with open('silhoutte_score.txt', 'a') as f:
        for n_clusters in range_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=10, n_init="auto")
            labels = kmeans.fit_predict(reduced_data[['PC1', 'PC2']])
            sum_of_squared_distances.append(kmeans.inertia_)
            score = silhouette_score(reduced_data[['PC1', 'PC2']], labels)
            silhouette_avg.append((score, n_clusters))
            f.write(f"For n_clusters = {n_clusters} The average silhouette_score is : {score}\n")
    
        
    n_clusters = max(silhouette_avg, key=lambda x: x[0])[1]
    kmeans = KMeans(n_clusters=n_clusters, random_state=10, n_init="auto").fit(reduced_data[['PC1', 'PC2']])

    return kmeans.labels_

def create_df_pca(df, columns, keyword):

    df_new = pd.DataFrame()
    df_new['Name'] = df['Name']
    fxc_to_drop = []
    
    for column in columns:
        if keyword in column:
            fxc = column.split('-DFT-')[0]
            df_new[fxc] = df[column]
            if '-D3' in fxc or '-D4' in fxc:
                    fxc_to_drop.append(fxc)
    
    #df_new.drop(columns=fxc_to_drop, inplace=True)
    df_new.drop(columns=['MP2', 'HF', 'HFS'], inplace=True)
    #df_new.drop(columns=['VWN3', 'PWLDA', 'VWN'], inplace=True)
    df_new.set_index('Name', inplace=True)
    
    return df_new.T

def corr_extrapolation(reference_values, old_reference_values):

    df_ref = pd.read_csv(reference_values, sep="\s+")
    df_ref_old = pd.read_csv(old_reference_values, sep="\s+")
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))

    plot_correlation(df_ref, df_ref_old, 'FORWARD', axes[0])
    plot_correlation(df_ref, df_ref_old, 'REVERSE', axes[1])
    plot_correlation(df_ref, df_ref_old, 'REACTION', axes[2])

    fig.text(0.5, 0.04, 'DLPNO-CCSD(T) extrapolation DZ-->TZ', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'DLPNO-CCSD(T) extrapolation TZ-->QZ', va='center', rotation='vertical', fontsize=12)

    plt.tight_layout(rect=[0.05, 0.05, 1, 1])

    plt.savefig('correlation_extrapolation.pdf', dpi=300)

def plot_correlation(df1, df2, column, ax):
    mae, rmse, r2 = calc_metrics(df1, df2, column)

    # Scatter plot with black border around points
    sns.scatterplot(y=df1[column], x=df2[column], ax=ax, edgecolor="black", s=50, color="darkmagenta")

    # Line y=x for correlation visualization
    lims = [min(df1[column].min(), df2[column].min()), max(df1[column].max(), df2[column].max())]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

    # Display metrics on the plot
    ax.text(0.02, 0.98, f'MAE: {mae:.2f} kcal/mol', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.02, 0.90, f'RMSE: {rmse:.2f} kcal/mol', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.02, 0.82, f'R²: {r2:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')

    # Set titles and labels based on column name
    if column == 'FORWARD':
        ax.set_title('$\Delta$E$^{\ddag}$$_{forward}$')
    elif column == 'REVERSE':
        ax.set_title('$\Delta$E$^{\ddag}$$_{reverse}$')
    elif column == 'REACTION':
        ax.set_title('$\Delta$E$_{reaction}$')

    # Remove individual x and y labels for subplots
    ax.set_xlabel('')
    ax.set_ylabel('')

def calc_metrics(df1, df2, column):
    mae = mean_absolute_error(df1[column], df2[column])
    rmse = np.sqrt(mean_squared_error(df1[column], df2[column]))
    r2 = r2_score(df1[column], df2[column])
    return mae, rmse, r2

def std_plot(full_data, class_data, full_data_bh9, class_data_bh9):

    df_fx_cyclo = pd.read_csv(full_data)
    df_classes_cyclo = pd.read_csv(class_data, sep=';')
    df_fx_cyclo['Type'] = df_classes_cyclo['classes']
    dict_replace = {'3_2': '[3+2] cycloaddition', 
                    'DA': '[4+2] cycloaddition',
                    '3_3': '[3,3] rearrangement',}
    df_fx_cyclo = df_fx_cyclo.replace(dict_replace)

    df_fx_cyclo_truncated = pd.read_csv("cyclo70_dh/full_data.csv")
    df_classes_cyclo = pd.read_csv(class_data, sep=';')
    df_fx_cyclo_truncated['Type'] = df_classes_cyclo['classes']
    dict_replace = {'3_2': '[3+2] cycloaddition', 
                    'DA': '[4+2] cycloaddition',
                    '3_3': '[3,3] rearrangement',}
    df_fx_cyclo_truncated = df_fx_cyclo_truncated.replace(dict_replace)

    df_fx_bh9 = pd.read_csv(full_data_bh9)
    df_classes_bh9 = pd.read_csv(class_data_bh9, sep=';')
    df_fx_bh9['Type'] = df_classes_bh9['classes']
    df_fx_bh9 = df_fx_bh9.loc[df_fx_bh9['Type'].isin(['DA', '3_2', '3_3'])]
    dict_replace = {'3_2': '[3+2] cycloaddition', 
                    'DA': '[4+2] cycloaddition',
                    '3_3': '[3,3] rearrangement',}
    df_fx_bh9 = df_fx_bh9.replace(dict_replace)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    g_1 = sns.kdeplot(data=df_fx_cyclo, x="Std_DFT_forward", fill=True, color='navy', label="Cyclo70")
    g_1 = sns.kdeplot(data=df_fx_bh9, x="Std_DFT_forward", fill=True, color='darkred', label="BH9")


    axs.set_ylabel('Density')
    axs.margins(x=0, y=0)
    axs.set_xlim(0.0, 20.0)
    axs.set_ylim(0.0, 0.40)

    axs.set_xlabel('$\sigma$ ($\Delta$E$^{\ddag}$$_{forward}$) (kcal mol$^{-1}$)')

    axs.tick_params(axis='both', which='both', length=0)

    axs.text(0.70, 0.88,
                f"$\mu$ Cyclo70: {df_fx_cyclo['Std_DFT_forward'].mean():.2f}" + ' kcal mol$^{-1}$\n', transform=axs.transAxes, fontsize=12, color='navy')
    axs.text(0.70, 0.80,
                f"$\mu$ BH9: {df_fx_bh9['Std_DFT_forward'].mean():.2f}" + ' kcal mol$^{-1}$\n', transform=axs.transAxes, fontsize=12, color='darkred')

    plt.legend()
    plt.tight_layout()
    plt.savefig('std_91.png', dpi=300)
    plt.savefig('std_91.pdf', dpi=300)
    
    plt.clf()

    # Define labels for legend
    labels = ["Cyclo70", "BH9"]
    colors = ["midnightblue", "darkred"]

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))

    for i, type in enumerate(['[3+2] cycloaddition', '[4+2] cycloaddition', '[3,3] rearrangement']):
        df_1 = df_fx_cyclo.loc[df_fx_cyclo['Type'] == type]
        df_2 = df_fx_bh9.loc[df_fx_bh9['Type'] == type]


        g_1 = sns.kdeplot(data=df_1, x="Std_DFT_forward", fill=True, color=colors[0], ax=axs[i], label=labels[0])
        g_1 = sns.kdeplot(data=df_2, x="Std_DFT_forward", fill=True, color=colors[1], ax=axs[i], label=labels[1])

        axs[i].set_title(f'Reaction class: {type}', fontsize=12)
        axs[i].margins(x=0, y=0)
        axs[i].set_xlim(0.0, 17.5)
        axs[i].set_ylim(0.0, 0.40)
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')
    
    fig.text(0.5, 0.02, '$\sigma$ ($\Delta$E$^{\ddag}$$_{forward}$) (kcal mol$^{-1}$)', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Density', va='center', rotation='vertical', fontsize=14)

    axs[0].legend(loc="upper right", fontsize=12)
        

    #axs.set_ylabel('Density')
    
    #axs.set_xlabel('$\sigma$ ($\Delta$E$^{\ddag}$) (kcal mol$^{-1}$)')
    #sns.move_legend(g_1, loc='upper left')
    #axs.tick_params(axis='both', which='both', length=0)

    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    plt.savefig('std_91_type.png', dpi=300)
    plt.savefig('std_91_type.pdf', dpi=300)

def std_plot_jctc(std_surrogate_model, std_bh9, std_cyclo70):

    df_cyclo = pd.read_csv(std_cyclo70)

    df_bh9 = pd.read_csv(std_bh9, sep=';')
    df_bh9 = df_bh9.loc[df_bh9['Type'].isin(['Diels-Alder', '[3+2]cycloaddition', '[3,3]rearrangement'])]

    df_full = pd.read_csv(std_surrogate_model, sep=';')
    
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    g_1 = sns.kdeplot(data=df_cyclo, x="Std_DFT_forward", fill=True, color='navy', label="Cyclo70")
    g_1 = sns.kdeplot(data=df_bh9, x="Std_DFT_forward", fill=True, color='darkred', label="BH9")
    g_1 = sns.kdeplot(data=df_full, x="prediction 4923", fill=True, color='darkgreen', label='Chemical space')

    axs.set_ylabel('Density')
    axs.margins(x=0, y=0)
    axs.set_xlim(0.0, 12.0)
    axs.set_ylim(0.0, 0.40)

    axs.set_xlabel('$\sigma$ ($\Delta$E$^{\ddag}$$_{forward}$) (kcal mol$^{-1}$)')

    axs.tick_params(axis='both', which='both', length=0)

    axs.text(0.65, 0.72,
                f"$\mu$ Cyclo70: {df_cyclo['Std_DFT_forward'].mean():.2f}" + ' kcal mol$^{-1}$\n', transform=axs.transAxes, fontsize=12, color='navy')
    axs.text(0.65, 0.80,
                f"$\mu$ BH9: {df_bh9['Std_DFT_forward'].mean():.2f}" + ' kcal mol$^{-1}$\n', transform=axs.transAxes, fontsize=12, color='darkred')
    axs.text(0.65, 0.88,
                f"$\mu$ Chemical space: {df_full['prediction 4923'].mean():.2f}" + ' kcal mol$^{-1}$\n', transform=axs.transAxes, fontsize=12, color='darkgreen')

    plt.legend()
    plt.tight_layout()
    plt.savefig('std_20_jctc.png', dpi=300)
    plt.savefig('std_20_jctc.pdf', dpi=300)
    

#### additional


def correlation_eric(maes_cyclo70, maes_bh9):
    
    df_mae_bh9 = pd.read_csv(maes_bh9)
    df_mae_cyclo70 = pd.read_csv(maes_cyclo70)
    df_mae_cyclo70 = df_mae_cyclo70.loc[df_mae_cyclo70['Methods'].isin(df_mae_bh9['Methods'])]
    
    df_mae_bh9['Subset'] = ['BH9'] * len(df_mae_bh9)
    df_mae_cyclo70['Subset'] = ['Cyclo70'] * len(df_mae_cyclo70)

    subset_functionals = ['PBE-QIDH', 'kPr2SCAN50', 'revDOD-PBEP86-D4_2021', 'wB97X-2', 'wB2PLYP18', 'wB97M-V', 'wB97M-D4rev', 'wB97M-D3BJ', 'r2SCAN50', 'M062X', 'r2SCAN0']
    
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    sns.scatterplot(y=df_mae_cyclo70['MAE reaction'], x=df_mae_bh9['MAE reaction'], edgecolor="black", s=45, color="darkmagenta")

    # Line y=x for correlation visualization
    lims = [min(df_mae_cyclo70['MAE reaction'].min(), df_mae_bh9['MAE reaction'].min()), max(df_mae_cyclo70['MAE reaction'].max(), df_mae_bh9['MAE reaction'].max())]
    axes.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    axes.set_ylabel('MAD $\Delta_r$E Cyclo70 (kcal mol$^{-1}$)')
    axes.set_xlabel('MAD $\Delta_r$E BH9 (kcal mol$^{-1}$)')
    plt.ylim((0, 20))
    plt.xlim((0, 20))
    ticks = np.arange(0, 21, 5)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.tight_layout()
    plt.savefig('correlation_re_eric.png', dpi=450)

    df_pics = pd.concat([df_mae_cyclo70, df_mae_bh9], ignore_index=True)
    fig, axes = plt.subplots(1, 1, figsize=(15, 6))

    #df_pics = df_pics.loc[df_pics['Methods'].isin(subset_functionals)]

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="white")
    plot = sns.barplot(x='Methods', y='MAE barrier', hue='Subset', data=df_pics, palette='dark')
    plt.ylabel('MAD $\Delta$E$^{\ddag}$ (kcal mol$^{-1}$)')
    plt.ylim((0, 14))
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig('histogram.png', dpi=450)


if __name__ == "__main__":

    args = parser.parse_args()

    if args.std:
        read_dat_files_to_dataframes(args.raw_data)
        combine_and_compute_std(args.raw_data)
    
    if args.error:
        calculate_mae(args.reference_value, args.full_data, 'complementary_files/methods_exx.csv')
    
    if args.pca:
        pca_plots(args.reference_value, args.full_data, args.rung_data, 'mae_ordered_BH.csv')

    if args.corr_reference:
        corr_extrapolation(args.reference_value, args.old_reference_value)

    if args.std_plot:
        std_plot(args.full_data, args.class_data, args.full_data_bh9, args.class_data_bh9)

    if args.std_plot_jctc:
       std_plot_jctc('jctc_data/chemical_space_ucb_9.csv',  'jctc_data/data_smiles_curated.csv', 'jctc_data/final_overview_data_8.csv')
    
    
    #calculate_mae('reference_values/references_bh9.dat', 'bh9/full_data.csv', args.rung_data)
    correlation_eric('mae/mae_ordered_BH.csv', 'mae/mae_ordered_BH_BH9.csv')
    
    
    
    
    
