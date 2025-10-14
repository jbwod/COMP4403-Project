import ipywidgets as widgets
from IPython.display import display, clear_output

# Global var for scenario data
scenario_data = {
    'type': 'normal',
    'rips': []
}

def create_widgets():
    """Create scenario configuration widgets."""
    # Scenario type selector
    scenario_type = widgets.Dropdown(
        options=[
            ('Normal', 'normal'),
            ('Kill Peers', 'kill_peers'),
        ],
        value='normal',
        description='Scenario:',
        style={'description_width': 'initial'}
    )
    
    #  text input
    rip_list = widgets.Textarea(
        value='',
        placeholder='Enter kill/revive as an array:\n[{"id": 1, "kill_r": 5, "rev_r": 10}, {"id": 2, "kill_r": 8, "rev_r": null}]',
        description='Node RIP List:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px', height='100px')
    )
    
    # Set scenario button
    set_btn = widgets.Button(
        description='Set Scenario',
        button_style='success',
        layout=widgets.Layout(width='200px')
    )
    
    # Output area
    output_area = widgets.Output()
    
    return (scenario_type, rip_list, set_btn, output_area)

def on_set_clicked(b, scenario_type, rip_list, output_area):
    """Handle set scenario button click."""
    global scenario_data
    
    b.description = "Setting..."
    b.disabled = True
    
    try:
        with output_area:
            clear_output(wait=True)
            print("Setting scenario...")
            print(f"Selected scenario: {scenario_type.value}")
            
            if scenario_type.value == 'kill_peers':
                import json
                try:
                    rips = json.loads(rip_list.value) if rip_list.value.strip() else []
                    print(f"Node RIP list: {rips}")
                    scenario_data = {
                        'type': 'kill_peers',
                        'rips': rips
                    }
                    print("Kill peers scenario set successfully!")
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    print("Please enter valid JSON format")
            else:
                scenario_data = {
                    'type': 'normal',
                    'rips': []
                }
                print("Normal scenario set successfully!")
            
    except Exception as e:
        print(f"Error setting scenario: {e}")
    finally:
        b.description = "Set Scenario"
        b.disabled = False

def display_scenario_widgets():
    """Display the scenario widgets."""

    (scenario_type, rip_list, set_btn, output_area) = create_widgets()
    
    def on_set_click(b):
        on_set_clicked(b, scenario_type, rip_list, output_area)
    
    set_btn.on_click(on_set_click)
    
    kill_peers_config = widgets.VBox([
        widgets.HTML("<h3>Kill Peers Configuration</h3>"),
        widgets.HTML("""
        <code>[{"id": node_id, "kill_r": kill_round, "rev_r": revive_round|null}]</code><br><br>
        """),
        rip_list
    ])
    
    def update_config_visibility(change):
        if change['new'] == 'kill_peers':
            kill_peers_config.layout.display = 'block'
        else:
            kill_peers_config.layout.display = 'none'
    
    kill_peers_config.layout.display = 'none' if scenario_type.value != 'kill_peers' else 'block'
    
    scenario_type.observe(update_config_visibility, names='value')
    
    # Layouts
    controls = widgets.VBox([
        widgets.HTML("<h2>Scenario Configuration</h2>"),
        widgets.HBox([scenario_type, set_btn]),
        kill_peers_config,
        widgets.HTML("<hr>"),
        output_area
    ])
    
    display(controls)
    print("Configure your scenario and click 'Set Scenario' to apply.")

def get_scenario_data():
    """Get the current scenario data."""
    return scenario_data
