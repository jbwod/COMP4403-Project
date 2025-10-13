import matplotlib.pyplot as plt
from src.graph import nxgraph
import ipywidgets as widgets
from src.analytics import clustering_co, path_length
from utils.plotter import draw_graph
from widgets.init_widget import set_graph_data
import src.agent as agent
from utils.simulator_new import simulate_round_agent_driven, get_network_stats
from utils.plotter import draw_graph, draw_gossip_step_by_step



def weighted_scene(seed,nodes,lower_ut,upper_ut,FILE_PIECES,n_seeders,search_mode,neighbor_selection,ttl,single_agent):
    # Scenario: Changes in agent behaviour with weighted and unweighted graphs
    G = nxgraph() 
    # Scenario parameters
    seed             = seed
    nodes            = nodes
    lower_ut         = lower_ut
    upper_ut         = upper_ut
    FILE_PIECES      = FILE_PIECES
    n_seeders        = n_seeders
    sim_seed         = seed
    search_mode      = search_mode
    neighbor_selection = neighbor_selection
    ttl              = ttl
    cleanup_queries  = True
    single_agent     = single_agent
    save_images      = False
    visualize_output = True
    output_area      = widgets.Output()    

    # Generating graphs
    weight_test_data = {
    'BA': [],
    'ER': []
    }

    # Generating BA graph
    for m in range(1, 5, 1):
        ba = G.BA_graph(
            nodes    = nodes,
            edges    = m,
            seed     = seed,
            weighted = True,
            lower_ut = lower_ut,
            upper_ut = upper_ut
        )

        node_count = ba.number_of_nodes()
        edge_count = ba.number_of_edges()
        avg_deg    = sum(dict(ba.degree()).values()) / node_count
        cluster    = round(clustering_co(ba), 2)
        path_L     = round(path_length(ba), 2)


        weight_test_data['BA'].append({
        'graph'                 : ba,
        'num_nodes'             : node_count,
        'num_edges'             : edge_count,
        'avg_degree'            : avg_deg,
        'clustering_coefficient': cluster,
        'average_path_length'   : path_L,
        'total_queries'         : 0,
        'total_hits'            : 0,
        'total_transfers'       : 0,
        'final_round'           : 0
        })

        print(f"\nBA_graph n={nodes}, m={m}")
        draw_graph(ba, edge_labels=None, total_pieces=FILE_PIECES)
        print(f"  | avg_deg={avg_deg:.2f}, C={cluster}, L={path_L}, edges={edge_count}")

    # Generating ER graph
    for ba_rec in weight_test_data['BA']:
        n       = ba_rec['num_nodes']
        avg_deg = ba_rec['avg_degree']
        m_er    = int(n * avg_deg / 2)

        er = G.ER_Graph_nm(
            nodes    = nodes,
            edges    = m_er,
            weighted = True,
            seed     = seed,
            lower_ut = lower_ut,
            upper_ut = upper_ut
        )

        node_count = er.number_of_nodes()
        edge_count = er.number_of_edges()
        cluster    = round(clustering_co(er), 2)
        path_L     = round(path_length(er), 2)

        weight_test_data['ER'].append({
        'graph'                 : er,
        'num_nodes'             : node_count,
        'num_edges'             : edge_count,
        'avg_degree'            : avg_deg,
        'clustering_coefficient': cluster,
        'average_path_length'   : path_L,
        'total_queries'         : 0,
        'total_hits'            : 0,
        'total_transfers'       : 0,
        'final_round'           : 0
        })
        print(f"\nER_graph n={nodes}, m={m_er}")
        draw_graph(er, edge_labels=None, total_pieces=FILE_PIECES)
        print(f"  | avg_deg={avg_deg:.2f}, C={cluster}, L={path_L}, edges={edge_count}")

    # running simulation
    for kind in ('BA', 'ER'):
        for rec in weight_test_data[kind]:
            Gk       = rec['graph']
            rec_q    = 0
            rec_h    = 0
            rec_t    = 0

            # initialise simulation
            set_graph_data(Gk, FILE_PIECES)
            agent.assign_n_seeders(Gk, n=n_seeders, seed=sim_seed)
            agent.initialize_file_sharing(
                Gk, FILE_PIECES,
                seed              = sim_seed,
                distribution_type = 'n_seeders',
                n_seeders         = n_seeders
            )
            # simulating
            r = 0
            while r <= 1000:
                result = simulate_round_agent_driven(
                Gk,
                FILE_PIECES,
                seed                    = sim_seed + r,
                cleanup_completed_queries=cleanup_queries,
                search_mode             = search_mode,
                current_round           = r,
                neighbor_selection      = neighbor_selection,
                single_agent            = (single_agent or None)
                )

                queries, hits, _ = result['message_rounds']
                transfers       = result['transfers']

                rec_q += len(queries)
                rec_h += len(hits)
                rec_t += len(transfers)
                stats = get_network_stats(Gk, FILE_PIECES)
                print(f"\n{kind} n={rec['num_nodes']} → Queries = {rec_q}, Hits = {rec_h}, Transfers = {rec_t}, Current round = {r}")
                print(f"Seeders: {stats['seeders']}")
                print(f"Leechers: {stats['leechers']} (incomplete: {stats['incomplete_leechers']})")
                print(f"Hybrids: {stats['hybrids']} (incomplete: {stats['incomplete_hybrids']})")
                print(f"Total pieces in network: {stats['total_pieces_in_network']}")
                print('about to go into stats')
                if stats['completion_rate'] >= 1.0:
                    print(f"\nAll nodes have all pieces in {r} rounds")
                    rec['final_round'] = r
                    break

                r += 1
                
            rec['total_queries']   = rec_q
            rec['total_hits']      = rec_h
            rec['total_transfers'] = rec_t

            print(f"\n{kind} n={rec['num_nodes']} → Queries = {rec_q}, Hits = {rec_h}, Transfers = {rec_t}")
            draw_gossip_step_by_step(Gk, result['message_rounds'], result['transfers'], 
                            FILE_PIECES, r, save_images=False, 
                            max_ttl=ttl, show_debug_info=False)
    # result evaluations

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['total_queries', 'total_hits', 'total_transfers', 'final_round']
    titles  = ['Total Queries vs Edges', 'Total Hits vs Edges',
           'Total Transfers vs Edges', 'Final Round vs Edges']
    colors  = ['blue', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'd']

    for i, metric in enumerate(metrics):
        row, col = divmod(i, 2)
        ax = axs[row][col]

        for kind in ['BA', 'ER']:
            edges = [rec['num_edges'] for rec in weight_test_data[kind]]
            values = [rec[metric] for rec in weight_test_data[kind]]
            ax.plot(edges, values, marker=markers[i], label=kind, color=colors[i] if kind == 'BA' else 'gray')

        ax.set_title(titles[i])
        ax.set_xlabel('Number of Edges')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    return weight_test_data
