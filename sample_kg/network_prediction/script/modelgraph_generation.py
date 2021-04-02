import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.pyplot as ticker
import collections
import argparse
import numpy as np
import powerlaw
import thinkplot
import thinkstats2 as ts

def main():
    start_time = time.time()
    
    ## Generate model graph ##
    if args.graph == 'er': # Erdős-Rényi model, Random graph
        n = 5000 # node
        p = 0.04 # probability of edge generation
        seed = 1234
        print(f'Graph type: Erdős-Rényi model\n'
              f'# n: {n}\n'
              f'# p: {p}\n'
              f'# seed: {seed}\n'
              f'Generating graph...')
        G = nx.fast_gnp_random_graph(n=n, p=p, seed=seed, directed=False)

    elif args.graph == 'ws': # Watts–Strogatz model, Small-world graph
        n = 5000 # node
        k = 20 # the number of neighbor node to connect with respect to every node
        p = 0.01 # re-wiring probability. Generate random graph when p=1.
        seed =1234
        print(f'Graph type: Watts–Strogatz model\n'
              f'# n: {n}\n'
              f'# k: {k}\n'
              f'# p: {p}\n'
              f'# seed: {seed}\n'
              f'Generating graph...')
        G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)

    elif args.graph == 'ba': # Barabási–Albert model, Scale-free graph
        n = 5000 # node
        m = 10 # the number of new edge to wire with the existing nodes
        seed = 1234
        print(f'Graph type: Barabási–Albert model\n'
              f'# n: {n}\n'
              f'# m: {m}\n'
              f'# seed: {seed}\n'
              f'Generating graph...')
        G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
 
    else:
        print('[ERROR] You need to select model graph.')


    ## Show graph summary ##
    print(f"-- graph summary --\n"
          f"# nodes: {nx.number_of_nodes(G)}\n"
          f"# edges: {nx.number_of_edges(G)}\n"
          f"# connected components: {nx.number_connected_components(G)}\n"
          f"# average shortest path: {nx.average_shortest_path_length(G)}\n"
          f"# average clustering coefficient: {nx.average_clustering(G)}\n"
          f"# degree assortativity coefficient: {nx.degree_assortativity_coefficient(G)}\n"
          f"# graph diameter: {nx.diameter(G)}\n"
          f"# graph density: {nx.density(G)}")

    ## Calculate average degree ##
    deg = []
    for k,l in G.degree(): # {node: degree, ...}
        deg.append(l)
    average_degree = sum(deg)/len(deg)
    print(f'# average degree: {average_degree}')

    ## Export generated graph as tsv file ##
    edge_type = "interaction" # Tentative edge type name
    with open(args.output, "w") as fp:
        for e in G.edges():
            fp.write(str(e[0]) + "\t" + edge_type + "\t" + str(e[1]) + "\n")
    print(f'[SAVE] graph file: {args.output}')


    ## Calculate degree distribution probability ##
    pmf = ts.Pmf(deg) # {degree: probability, ...}
    # print(f'pmf mean: {pmf.Mean()}, pmf std: {pmf.Std()}')
    pmf_degree = [] # degree
    pmf_prob = [] # degree distribution probability
    for i in pmf:
        pmf_degree.append(i)
        pmf_prob.append(pmf[i])


    ## power law fitting using mudule ##
    print(f'--- power law fitting parameter ---')
    np.seterr(divide='ignore', invalid='ignore') # a magical spell
    fit = powerlaw.Fit(deg, discrete=True) # fitting degree distribution probability to linear
    R, p = fit.distribution_compare('power_law', 'exponential')
    print(f'# power law gamma: {fit.power_law.alpha}\n'
        f'# gammma standard error(std): {fit.power_law.sigma}\n'
        f'# fix x min: {fit.fixed_xmin}\n'
        f'# discrete: {fit.discrete}\n'
        f'# x min: {fit.xmin}\n'
        f'# loglikelihood ratio: {R}\n'
        f'# significant value: {p}')


    ## Plot degree distbibution probability (normal scale) ##   
    fig = plt.figure(figsize=(8,6), tight_layout=True)
    ax = fig.add_subplot(1,1,1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)

    ax.scatter(pmf_degree, pmf_prob, c='black', s=30, alpha=1, linewidths=0)
    ax.tick_params(direction='out', which='major', axis='both', length=4, width=1, labelsize=20, pad=10)
    ax.set_xlabel('k', fontsize=25, labelpad=10)
    ax.set_ylabel('P(k)', fontsize=25, labelpad=10)
    deg_fig_name = args.fig + '_degdist_plot.pdf'
    plt.savefig(deg_fig_name, dpi=300, format='pdf', transparent=True)
    plt.clf()
    print(f'[SAVE] degree distribution figure: {deg_fig_name}')


    ## Plot degree distbibution probability (log scale) ##
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
    # ax.xaxis.set_minor_formatter(ticker.ScalarFormatter()) # Set x axis minot tick as integ, Activate only for WS
    ax.yaxis.set_minor_formatter(ticker.NullFormatter()) # Set y axis minor tick unvisible

    ax.scatter(pmf_degree, pmf_prob, c='black', s=30, linewidths=0) # plot probability of degree distribution
    if R > 0:
        fit.power_law.plot_pdf(c='#766AFF', linestyle='dotted', linewidth=2, alpha=1) # plot power law fitting

    ax.tick_params(direction='in', which='major', axis='both',
                     length=7, width=1, labelsize=20, pad=10) # Set major tick parameter
    ax.tick_params(direction='in', which='minor', axis='both',
                     length=4, width=1, labelsize=20, pad=10) # Set minor tick parameter
    ax.set_xlabel('k', fontsize=25, labelpad=10)
    ax.set_ylabel('P(k)', fontsize=25, labelpad=10)
    ymin = min(pmf_prob)
    ymin_ = pow(10, round(np.log10(ymin))) - pow(10, round(np.log10(ymin)-1))
    ax.set_ylim(ymin_,)
    log_fig_name = args.fig + '_Pk_plot.pdf'
    fig.savefig(log_fig_name, dpi=300, format='pdf', transparent=True)
    plt.clf()
    print(f'[SAVE] degree distribution figure (log-scale): {log_fig_name}')


    elapsed_time = time.time() - start_time
    print(f'[TIME]: {elapsed_time} sec')
    print(f'Completed!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help="graph type: er, ba, ws")
    parser.add_argument('--output', type=str, help="graph file name (XXXX.graph.tsv)")
    parser.add_argument('--fig', type=str, help="figure name")
    args = parser.parse_args()

    main()
