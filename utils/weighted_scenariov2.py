import matplotlib.pyplot as plt
from src.graph import nxgraph
import ipywidgets as widgets
from src.analytics import clustering_co, path_length
from utils.plotter import draw_graph
from widgets.init_widget import set_graph_data
import src.agent as agent
from utils.simulator_new import simulate_round_agent_driven, get_network_stats
from utils.plotter import draw_graph, draw_gossip_step_by_step

def weighted_scene(seed, nodes, lower_ut, upper_ut, FILE_PIECES, n_seeders, search_mode, neighbor_selection, ttl, single_agent):
    G = nxgraph()
    output_area = widgets.Output()

    def run_simulation(weighted_flag):
        graph_data = {'BA': [], 'ER': []}
        for m in range(1, 8):
            ba = G.BA_graph(
                nodes=nodes,
                edges=m,
                seed=seed,
                weighted=weighted_flag,
                lower_ut=lower_ut,
                upper_ut=upper_ut
            )
            node_count = ba.number_of_nodes()
            edge_count = ba.number_of_edges()
            avg_deg = sum(dict(ba.degree()).values()) / node_count
            cluster = round(clustering_co(ba), 2)
            path_L = round(path_length(ba), 2)

            graph_data['BA'].append({
                'graph': ba,
                'num_nodes': node_count,
                'num_edges': edge_count,
                'avg_degree': avg_deg,
                'clustering_coefficient': cluster,
                'average_path_length': path_L,
                'total_queries': 0,
                'total_hits': 0,
                'total_transfers': 0,
                'final_round': 0
            })

        for ba_rec in graph_data['BA']:
            n = ba_rec['num_nodes']
            avg_deg = ba_rec['avg_degree']
            m_er = int(n * avg_deg / 2)

            er = G.ER_Graph_nm(
                nodes=nodes,
                edges=m_er,
                weighted=weighted_flag,
                seed=seed,
                lower_ut=lower_ut,
                upper_ut=upper_ut
            )

            node_count = er.number_of_nodes()
            edge_count = er.number_of_edges()
            cluster = round(clustering_co(er), 2)
            path_L = round(path_length(er), 2)

            graph_data['ER'].append({
                'graph': er,
                'num_nodes': node_count,
                'num_edges': edge_count,
                'avg_degree': avg_deg,
                'clustering_coefficient': cluster,
                'average_path_length': path_L,
                'total_queries': 0,
                'total_hits': 0,
                'total_transfers': 0,
                'final_round': 0
            })

        for kind in ('BA', 'ER'):
            for rec in graph_data[kind]:
                Gk = rec['graph']
                rec_q = rec_h = rec_t = 0

                set_graph_data(Gk, FILE_PIECES)
                agent.assign_n_seeders(Gk, n=n_seeders, seed=seed)
                agent.initialize_file_sharing(
                    Gk, FILE_PIECES,
                    seed=seed,
                    distribution_type='n_seeders',
                    n_seeders=n_seeders
                )

                r = 0
                while r <= 1000:
                    result = simulate_round_agent_driven(
                        Gk,
                        FILE_PIECES,
                        seed=seed + r,
                        cleanup_completed_queries=True,
                        search_mode=search_mode,
                        current_round=r,
                        neighbor_selection=neighbor_selection,
                        single_agent=(single_agent or None)
                    )

                    queries, hits, _ = result['message_rounds']
                    transfers = result['transfers']

                    rec_q += len(queries)
                    rec_h += len(hits)
                    rec_t += len(transfers)
                    stats = get_network_stats(Gk, FILE_PIECES)

                    if stats['completion_rate'] >= 1.0:
                        rec['final_round'] = r
                        break
                    r += 1

                rec['total_queries'] = rec_q
                rec['total_hits'] = rec_h
                rec['total_transfers'] = rec_t

        return graph_data

    # Run both weighted and unweighted
    weighted_data = run_simulation(weighted_flag=True)
    unweighted_data = run_simulation(weighted_flag=False)

    # ─────────────────────────────────────────────────────────────
    # Side-by-side graph visualization
    # ─────────────────────────────────────────────────────────────
    def compare_graphs(kind):
        for i in range(len(weighted_data[kind])):
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].set_title(f"{kind} Weighted (Edges={weighted_data[kind][i]['num_edges']})")
            axs[1].set_title(f"{kind} Unweighted (Edges={unweighted_data[kind][i]['num_edges']})")

            draw_graph(weighted_data[kind][i]['graph'], edge_labels=None, total_pieces=FILE_PIECES)
            draw_graph(unweighted_data[kind][i]['graph'], edge_labels=None, total_pieces=FILE_PIECES)

            plt.tight_layout()
            plt.show()

    compare_graphs('BA')
    compare_graphs('ER')

    return weighted_data, unweighted_data