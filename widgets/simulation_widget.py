import ipywidgets as widgets
from IPython.display import display, clear_output

def create_widgets():
    """Create simulation configuration widgets."""
    # Simulation type selector
    simulation_type = widgets.Dropdown(
        options=[
            ('Run', 'run'),
        ],
        value='run',
        description='Simulation:',
        style={'description_width': 'initial'}
    )
    
    # Simulation parameters
    max_rounds = widgets.IntSlider(
        value=70,
        min=10,
        max=200,
        step=10,
        description='Max Rounds:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    seed = widgets.IntSlider(
        value=42,
        min=0,
        max=1000,
        step=1,
        description='Seed:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    search_mode = widgets.Dropdown(
        options=[
            ('Single', 'single'),
            ('Realistic', 'realistic')
        ],
        value='realistic',
        description='Search Mode:',
        style={'description_width': 'initial'}
    )
    
    neighbor_selection = widgets.Dropdown(
        options=[
            ('Bandwidth', 'bandwidth'),
            ('Random', 'random')
        ],
        value='random',
        description='Neighbor Selection:',
        style={'description_width': 'initial'}
    )
    
    ttl = widgets.IntSlider(
        value=5,
        min=1,
        max=10,
        step=1,
        description='TTL:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    k = widgets.IntSlider(
        value=3,
        min=1,
        max=10,
        step=1,
        description='K (neighbors):',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    cleanup_queries = widgets.Checkbox(
        value=True,
        description='Cleanup Completed Queries',
        style={'description_width': 'initial'}
    )
    
    single_agent = widgets.IntText(
        value=0,
        min=0,
        max=100,
        description='Single Agent (0=all):',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    save_images = widgets.Checkbox(
        value=False,
        description='Save Images',
        style={'description_width': 'initial'}
    )
    
    debug_output = widgets.Checkbox(
        value=False,
        description='Debug Output',
        style={'description_width': 'initial'}
    )
    
    # Run simulation button
    run_btn = widgets.Button(
        description='Run Simulation',
        button_style='success',
        layout=widgets.Layout(width='200px')
    )
    
    # Output area
    output_area = widgets.Output()
    
    return (simulation_type, max_rounds, seed, search_mode, neighbor_selection, 
            ttl, k, cleanup_queries, single_agent, save_images, debug_output, run_btn, output_area)

def on_run_clicked(b, simulation_type, max_rounds, seed, search_mode, neighbor_selection, 
                  ttl, k, cleanup_queries, single_agent, save_images, debug_output, output_area):
    """Handle run simulation button click."""
    b.description = "Running..."
    b.disabled = True
    
    try:
        with output_area:
            clear_output(wait=True)
            print("Running simulation...")
            print(f"Selected simulation: {simulation_type.value}")
            print(f"Max rounds: {max_rounds.value}")
            print(f"Seed: {seed.value}")
            print(f"Search mode: {search_mode.value}")
            print(f"Neighbor selection: {neighbor_selection.value}")
            print(f"TTL: {ttl.value}")
            print(f"K (neighbors): {k.value}")
            print(f"Cleanup queries: {cleanup_queries.value}")
            print(f"Single agent: {single_agent.value if single_agent.value > 0 else 'All agents'}")
            print(f"Save images: {save_images.value}")
            print(f"Debug output: {debug_output.value}")
            print("Note: This is a placeholder")
            
    except Exception as e:
        print(f"Error running simulation: {e}")
    finally:
        b.description = "Run Simulation"
        b.disabled = False

def display_simulation_widgets():
    """Display the simulation widgets."""

    (simulation_type, max_rounds, seed, search_mode, neighbor_selection, 
     ttl, k, cleanup_queries, single_agent, save_images, debug_output, run_btn, output_area) = create_widgets()
    
    def on_run_click(b):
        on_run_clicked(b, simulation_type, max_rounds, seed, search_mode, neighbor_selection, 
                      ttl, k, cleanup_queries, single_agent, save_images, debug_output, output_area)
    
    run_btn.on_click(on_run_click)
    
    # Layouts
    controls = widgets.VBox([
        widgets.HTML("<h2>Simulation Control</h2>"),
        widgets.HBox([simulation_type, run_btn]),
        widgets.HTML("<hr>"),
        widgets.HTML("<h3>Simulation Parameters</h3>"),
        widgets.HBox([max_rounds, seed]),
        widgets.HBox([search_mode, neighbor_selection]),
        widgets.HBox([ttl, k]),
        widgets.HBox([cleanup_queries, single_agent]),
        widgets.HBox([save_images, debug_output]),
        widgets.HTML("<hr>"),
        output_area
    ])
    
    display(controls)
    print("## PLACEHOLDER ##.")
