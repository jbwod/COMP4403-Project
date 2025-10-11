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
    
    # Run simulation button
    run_btn = widgets.Button(
        description='Run Simulation',
        button_style='success',
        layout=widgets.Layout(width='200px')
    )
    
    # Output area
    output_area = widgets.Output()
    
    return (simulation_type, run_btn, output_area)

def on_run_clicked(b, simulation_type, output_area):
    """Handle run simulation button click."""
    b.description = "Running..."
    b.disabled = True
    
    try:
        with output_area:
            clear_output(wait=True)
            print("Running simulation...")
            print(f"Selected simulation: {simulation_type.value}")
            print("Note: This is a placeholder")
            
    except Exception as e:
        print(f"Error running simulation: {e}")
    finally:
        b.description = "Run Simulation"
        b.disabled = False

def display_simulation_widgets():
    """Display the simulation widgets."""

    (simulation_type, run_btn, output_area) = create_widgets()
    
    def on_run_click(b):
        on_run_clicked(b, simulation_type, output_area)
    
    run_btn.on_click(on_run_click)
    
    # Layouts
    controls = widgets.VBox([
        widgets.HTML("<h2>Simulation Control</h2>"),
        widgets.HBox([simulation_type, run_btn]),
        widgets.HTML("<hr>"),
        output_area
    ])
    
    display(controls)
    print("## PLACEHOLDER ##.")
