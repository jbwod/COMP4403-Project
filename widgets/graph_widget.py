import ipywidgets as widgets
from IPython.display import display, clear_output
from src.graph import nxgraph
import src.agent as agent_module
from utils.plotter import draw_graph

graph_generated = False
G = None
FILE_PIECES = None

def create_widgets():
    graph_type = widgets.Dropdown(
        options=[('Barabasi-Albert (Scale-free)', 'BA'), ('Erdos-Renyi (Random)', 'ER')],
        value='BA',
        description='Graph Type:',
        style={'description_width': 'initial'}
    )
    
    num_nodes = widgets.IntSlider(
        value=12,
        min=5,
        max=100,
        step=1,
        description='Nodes:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    ba_edges = widgets.IntSlider(
        value=2,
        min=1,
        max=10,
        step=1,
        description='BA Edges:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    er_prob = widgets.FloatSlider(
        value=0.1,
        min=0.01,
        max=0.5,
        step=0.01,
        description='ER Probability:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    lower_bandwidth = widgets.IntSlider(
        value=10,
        min=1,
        max=50,
        step=1,
        description='Min Bandwidth:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    upper_bandwidth = widgets.IntSlider(
        value=100,
        min=50,
        max=200,
        step=5,
        description='Max Bandwidth:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    random_seed = widgets.IntSlider(
        value=42,
        min=0,
        max=1000,
        step=1,
        description='Random Seed:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    num_seeders = widgets.IntSlider(
        value=1,
        min=1,
        max=100,
        step=1,
        description='Seeders:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    file_pieces = widgets.IntSlider(
        value=15,
        min=5,
        max=50,
        step=1,
        description='File Pieces:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    generate_btn = widgets.Button(
        description='Generate Graph',
        button_style='primary',
        layout=widgets.Layout(width='150px')
    )
    
    output_area = widgets.Output()
    
    return (graph_type, num_nodes, ba_edges, er_prob, lower_bandwidth, 
            upper_bandwidth, random_seed, num_seeders, file_pieces, 
            generate_btn, output_area)

def update_controls_visibility(graph_type, ba_edges, er_prob, change):
    """Update widget visibility based on graph type selection."""
    if change['new'] == 'BA':
        ba_edges.layout.display = 'flex'
        er_prob.layout.display = 'none'
        ba_edges.description = 'BA Edges:'
        er_prob.description = 'ER Probability: (inactive)'
    else:  # ER
        ba_edges.layout.display = 'none'
        er_prob.layout.display = 'flex'
        ba_edges.description = 'BA Edges: (inactive)'
        er_prob.description = 'ER Probability:'

def on_generate_clicked(b, graph_type, num_nodes, ba_edges, er_prob, 
                       lower_bandwidth, upper_bandwidth, random_seed, 
                       num_seeders, file_pieces, output_area):
    """Handle graph generation button click."""
    global graph_generated, G, FILE_PIECES
    
    b.description = "Generating..."
    b.disabled = True
    
    try:
        with output_area:
            clear_output(wait=True)
            print("Generating graph...")
            
            # Create new graph
            G = nxgraph()
            
            if graph_type.value == 'BA':
                G.BA_graph(
                    nodes=num_nodes.value, 
                    edges=ba_edges.value, 
                    seed=random_seed.value, 
                    weighted=True, 
                    lower_ut=lower_bandwidth.value, 
                    upper_ut=upper_bandwidth.value
                )
                print(f"Generated BA graph with {num_nodes.value} nodes, {ba_edges.value} edges per new node")
            else:  # ER
                G.ER_graph(
                    nodes=num_nodes.value, 
                    prob=er_prob.value, 
                    weighted=True, 
                    lower_ut=lower_bandwidth.value, 
                    upper_ut=upper_bandwidth.value
                )
                print(f"Generated ER graph with {num_nodes.value} nodes, p={er_prob.value}")
            
            agent_module.assign_n_seeders(G.graph, n=num_seeders.value, seed=random_seed.value)
            
            FILE_PIECES = file_pieces.value
            
            print(f"has {G.graph.number_of_nodes()} nodes and {G.graph.number_of_edges()} edges")
            print(f"has {num_seeders.value} seeders, {file_pieces.value} file pieces")
            
            draw_graph(G.graph, total_pieces=file_pieces.value)
            graph_generated = True
            
    except Exception as e:
        print(f"Error generating: {e}")
    finally:
        b.description = "Generate Graph"
        b.disabled = False

def display_graph_widgets():
    """Display the graph generation widgets."""
    global graph_generated, G, FILE_PIECES
    
    (graph_type, num_nodes, ba_edges, er_prob, lower_bandwidth, 
     upper_bandwidth, random_seed, num_seeders, file_pieces, 
     generate_btn, output_area) = create_widgets()
    
    def update_visibility(change):
        update_controls_visibility(graph_type, ba_edges, er_prob, change)
    
    graph_type.observe(update_visibility, names='value')
    update_visibility({'new': 'BA'})
    
    def on_click(b):
        on_generate_clicked(b, graph_type, num_nodes, ba_edges, er_prob, 
                           lower_bandwidth, upper_bandwidth, random_seed, 
                           num_seeders, file_pieces, output_area)
    
    generate_btn.on_click(on_click)

    # Layouts
    controls = widgets.VBox([
        widgets.HTML("<h1>Graph Configuration</h1>"),
        widgets.HBox([graph_type, num_nodes]),
        widgets.HBox([ba_edges, er_prob]),
        widgets.HBox([lower_bandwidth, upper_bandwidth]),
        widgets.HBox([random_seed, num_seeders]),
        widgets.HBox([file_pieces, generate_btn]),
        output_area
    ])
    
    display(controls)
    print("Use the controls above to configure and generate your graph.")
    print("Once you click 'Generate Graph', you can proceed to the next cell.")
    print("The graph will be available for use in subsequent cells.")

def get_graph():
    global graph_generated, G, FILE_PIECES
    if not graph_generated:
        raise RuntimeError("Graph must be generated before accessing. Call display_graph_widgets() and click 'Generate Graph' first.")
    return G, FILE_PIECES

def is_graph_generated():
    global graph_generated
    return graph_generated