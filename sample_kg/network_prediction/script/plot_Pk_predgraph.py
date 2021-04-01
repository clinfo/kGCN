# Author: yoshi
# Date: 01/25/2021
# Updated: 02/05/2021
# Project: NetworkPrediction
# Script: plot degree distribution
# Usage: python plot_degree_distribution_predicted_graph2.py --graphinput <score file> --th <threshold> --fig <figure name> --train --traininput <train graph file>

import networkx as nx
import pandas as pd
#import matplotlib as mpl # Add for server
#mpl.use('Agg') # add for server
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import collections
import argparse
import time
import thinkstats2 as ts
import powerlaw


def main():
    start_time = time.time()

    if args.train:
        ## include train dataset ##
        print(f'[LOAD] graph file: {args.graphinput}\n'
              f'[LOAD] train graph file: {args.traininput}\n'
              f'# top edges threshold: {args.th}')
        ## Load prediction score data ##
        score_table = pd.read_table(args.graphinput, sep='\t', header=0)
        score_table = score_table[score_table['train_edge'] == 0] # pick new/test edge exclusively
        testnew_table = score_table.iloc[:args.th,]
        ## Load train graph data ##
        train_table = pd.read_table(args.traininput, sep='\t', names=('gene1', 'gene2')) # [node1 node2]
        ## integrate test/new and train graphs ##
        graph_table = pd.concat([testnew_table, train_table], axis=0)
        print(f'# test+new table shape: {testnew_table.shape}\n'
              f'# train table shape: {train_table.shape}\n'
              f'# test+new+train table shape: {graph_table.shape}')

    else:
        print(f'[LOAD] graph input: {args.graphinput}\n'
              f'# top edges threshold: {args.th}')
        ## Load prediction score data
        score_table = pd.read_table(args.graphinput, sep='\t', header=0)
        score_table = score_table[score_table['train_edge'] == 0] # pick new/test edge exclusively
        graph_table = score_table.iloc[:args.th,]

    ## Load graph as undirected-nonmulti graph ##
    Gn = nx.from_pandas_edgelist(graph_table, source='gene1', target='gene2',
                                 edge_attr=None, create_using=nx.Graph())

    print(f"\n== integrated graph summary ==\n"
          f"# nodes: {nx.number_of_nodes(Gn)}\n"
          f"# edges: {nx.number_of_edges(Gn)}\n"
          f"# connected components: {nx.number_connected_components(Gn)}\n")

    ## Pick biggest component ##
    Gc = max(nx.connected_component_subgraphs(Gn), key=len)

    print(f"== biggest connected component graph summary ==\n"
          f"# nodes: {nx.number_of_nodes(Gc)}\n"
          f"# edges: {nx.number_of_edges(Gc)}\n"
          f"# connected components: {nx.number_connected_components(Gc)}\n"
          f"# average shortest path: {nx.average_shortest_path_length(Gc)}\n"
          f"# average clustering coefficient: {nx.average_clustering(Gc)}\n"
          f"# degree assortativity coefficient: {nx.degree_assortativity_coefficient(Gc)}\n"
          f"# graph diameter: {nx.diameter(Gc)}\n"
          f"# graph density: {nx.density(Gc)}")

    ## calculate degree distribution ##
    degree_seq = sorted([d for n, d in Gc.degree()], reverse=True) # G.degree()={node: degree, ...}
    average_degree = sum(degree_seq)/len(degree_seq)
    print(f'# average degree: {average_degree}\n')
    degreeCount = collections.Counter(degree_seq)
    deg, cnt = zip(*degreeCount.items()) # deg:degree per nodes, cnt: the number of each degree


    ## calculate probability mass function ##
    pmf = ts.Pmf(degree_seq) # {degree: probability, ...}
    # print(f'pmf mean: {pmf.Mean()}, pmf std: {pmf.Std()}')
    pmf_degree = [] # degree
    pmf_prob = [] # degree distribution probability
    for i in pmf:
        pmf_degree.append(i)
        pmf_prob.append(pmf[i])

    ## power law fitting ##
    print(f'== power law fitting parameter ==')
    np.seterr(divide='ignore', invalid='ignore') # a magical spell
    fit = powerlaw.Fit(degree_seq, discrete=True, xmin=1) # fitting degree distribution probability to linear
    R, p = fit.distribution_compare('power_law', 'exponential')
    print(f'# power law gamma: {fit.power_law.alpha}\n'
          f'# gammma standard error(std): {fit.power_law.sigma}\n'
          f'# fix x min: {fit.fixed_xmin}\n'
          f'# discrete: {fit.discrete}\n' 
          f'# x min: {fit.xmin}\n'
          f'# loglikelihood ratio: {R}\n'
          f'# significant value: {p}')


    ## make figure ##
    ## Plot degree distbibution (normal scale) ##
    fig = plt.figure(figsize=(8,6), tight_layout=True)
    ax = fig.add_subplot(1,1,1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)

    ax.scatter(deg, cnt, c='black', s=30, alpha=1, linewidth=0)
    ax.tick_params(direction='out', which='major', axis='both', length=4, width=1, labelsize=20, pad=10)
    ax.set_xlabel('degree', fontsize=25, labelpad=10)
    ax.set_ylabel('frequency', fontsize=25, labelpad=10)
    deg_fig_name = f'{args.fig}_top{args.th}_degdist_plot.pdf'
    fig.savefig(deg_fig_name, dpi=300, format='pdf', transparent=True)
    plt.clf()
    print(f'[SAVE] degree distribution figure: {deg_fig_name}')

    ## Plot degree distbibution (probability log scale) ##
    fig = plt.figure(figsize=(6,6), tight_layout=True)
    ax = fig.add_subplot(1,1,1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0)) # Set x axis major tick for log10
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0)) # Set y axis major tick for log10
    ax.xaxis.set_minor_formatter(ticker.NullFormatter()) # Set x axis minor tick unvisible
    ax.yaxis.set_minor_formatter(ticker.NullFormatter()) # Set y axis minor tick unvisible

    ax.scatter(pmf_degree, pmf_prob, c='black', s=30, alpha=1, linewidths=0) # plot probability of degree distribution
    if R > 0:
        fit.power_law.plot_pdf(c='#766AFF', linestyle='dotted', linewidth=2, alpha=1) # plot power law fitting
    ax.tick_params(direction='in', which='major', axis='both',
                    length=7, width=1, labelsize=20, pad=10) # Set major tick parameter
    ax.tick_params(direction='in', which='minor', axis='both',
                    length=4, width=1, labelsize=15, pad=10) # Set minor tick parameter
    ax.set_xlabel('k', fontsize=25, labelpad=10)
    ax.set_ylabel('P(k)', fontsize=25, labelpad=10)
    ymin = min(pmf_prob)
    ymin_ = pow(10, round(np.log10(ymin))) - pow(10, round(np.log10(ymin)-1))
    ax.set_ylim(ymin_,)
    Pk_fig_name = f'{args.fig}_top{args.th}_Pk_plot.pdf'
    fig.savefig(Pk_fig_name, dpi=300, format='pdf', transparent=True)
    plt.clf()
    print(f'[SAVE] P(k) degree distribution figure: {Pk_fig_name}')

    elapsed_time = time.time() - start_time
    print(f'[Time]: {elapsed_time} sec')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--graphinput', type=str, help="input data: prediction score txt file")
    parser.add_argument('--th', type=int, help="threthold: top N edges")
    parser.add_argument('--fig', type=str, help="figure name")
    parser.add_argument('--traininput', type=str, default= '', help="train input data: tsv file")
    parser.add_argument('--train', action='store_true', help="if trainset include or exclude")
    args = parser.parse_args()

    main()

