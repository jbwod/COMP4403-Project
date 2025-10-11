import ipywidgets as widgets
from IPython.display import display, clear_output
import src.agent as agent
from utils.simulator_new import reset_simulation
import random

G = None
FILE_PIECES = None

def set_graph_data(graph_obj, file_pieces):
    """Set graph data. Expects a NetworkX graph object."""
    global G, FILE_PIECES
    G = graph_obj
    FILE_PIECES = file_pieces

def get_graph_data():
    """Get the current graph and file pieces."""
    global G, FILE_PIECES
    return G, FILE_PIECES


def create_widgets():
    # Reset simulation button
    reset_btn = widgets.Button(
        description='Reset Simulation',
        button_style='warning',
        layout=widgets.Layout(width='200px')
    )
    
    # Distribution type selector
    distribution_type = widgets.Dropdown(
        options=[
            ('Assign N Full Seeders', 'n_seeders')
        ],
        value='n_seeders',
        description='Distribution:',
        style={'description_width': 'initial'}
    )
    
    # N seeders input - max is number of nodes in graph
    max_seeders = len(G.nodes()) if G is not None else 20
    n_seeders = widgets.IntSlider(
        value=1,
        min=1,
        max=max_seeders,
        step=1,
        description='N Seeders:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    # Seed for n-seeders
    n_seeders_seed = widgets.IntSlider(
        value=42,
        min=0,
        max=1000,
        step=1,
        description='Seed:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    
    # Apply button
    apply_btn = widgets.Button(
        description='Apply Distribution',
        button_style='success',
        layout=widgets.Layout(width='200px')
    )
    
    # Output area
    output_area = widgets.Output()
    
    return (reset_btn, distribution_type, n_seeders, n_seeders_seed, apply_btn, output_area)


def on_reset_clicked(b, output_area):
    """Handle reset simulation button click."""
    global G, FILE_PIECES
    
    if G is None or FILE_PIECES is None:
        with output_area:
            print("Error: No graph data available. Generate a graph first.")
        return
    
    b.description = "Resetting..."
    b.disabled = True
    
    try:
        with output_area:
            clear_output(wait=True)
            print("Resetting simulation...")
            
            reset_simulation(G, FILE_PIECES, seed=42)
            
            print("Simulation reset complete!")
            print("\nNode status after reset:")
            for node in G.nodes():
                info = agent.get_agent_info(G, node)
                print(f"Node {node}: {info['role']} - {info['num_pieces']}/{FILE_PIECES} pieces - Complete: {info['is_complete']}")
            
    except Exception as e:
        print(f"Error resetting simulation: {e}")
    finally:
        b.description = "Reset Simulation"
        b.disabled = False

def on_apply_clicked(b, n_seeders, n_seeders_seed, output_area):
    """Handle apply distribution button click."""
    global G, FILE_PIECES
    
    if G is None or FILE_PIECES is None:
        with output_area:
            print("Error: No graph data available. Generate a graph first.")
        return
    
    b.description = "Applying..."
    b.disabled = True
    
    try:
        with output_area:
            clear_output(wait=True)
            print("Applying file distribution...")
            

            agent.assign_n_seeders(G, n=n_seeders.value, seed=n_seeders_seed.value)
            agent.initialize_file_sharing(G, FILE_PIECES, seed=n_seeders_seed.value, 
                                               distribution_type='n_seeders', n_seeders=n_seeders.value)
            print(f"Assigned {n_seeders.value} full seeders with seed {n_seeders_seed.value}")
            
            print("\nNode status after distribution:")
            for node in G.nodes():
                info = agent.get_agent_info(G, node)
                print(f"Node {node}: {info['role']} - {info['num_pieces']}/{FILE_PIECES} pieces - Complete: {info['is_complete']}")
            
            print(f"\nTotal pieces in network: {sum(len(G.nodes[node].get('file_pieces', set())) for node in G.nodes())}")
            
    except Exception as e:
        print(f"Error applying distribution: {e}")
    finally:
        b.description = "Apply Distribution"
        b.disabled = False

def display_init_widgets():
    """Display the initialization widgets."""
    global G, FILE_PIECES
    
 
    (reset_btn, distribution_type, n_seeders, n_seeders_seed, apply_btn, output_area) = create_widgets()

    def on_reset_click(b):
        on_reset_clicked(b, output_area)
    
    def on_apply_click(b):
        on_apply_clicked(b, n_seeders, n_seeders_seed, output_area)
    
    reset_btn.on_click(on_reset_click)
    apply_btn.on_click(on_apply_click)
    
    # Layouts
    controls = widgets.VBox([
        widgets.HTML("<h2>Simulation Initialization</h2>"),
        widgets.HBox([reset_btn, widgets.HTML("<div style='width: 20px'></div>"), apply_btn]),
        widgets.HTML("<hr>"),
        widgets.HBox([distribution_type, n_seeders, n_seeders_seed]),
        widgets.HTML("<hr>"),
        output_area
    ])
    
    display(controls)
    print("Use the controls above to reset the simulation and apply file distributions.")
    print("Make sure to generate a graph first using the graph generation widgets.")
