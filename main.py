from dataclasses import dataclass, field  #%
from typing import Optional, List  #%
import random
import numpy as np
import graphviz
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import glob
import os

@dataclass
class Place:
    id: int
    name: Optional[str] = None
    tokens: int = 0

@dataclass
class Arc:
    weight: int
    place: Place

@dataclass
class Transition:
    id: int
    name: Optional[str] = None
    in_arcs: List[Arc] = field(default_factory=list)  #%
    out_arcs: List[Arc] = field(default_factory=list)  #%
    mu: float = 0.0  # Mean for Gaussian distribution
    sigma_squared: float = 1.0  # Variance for Gaussian distribution (sigma^2)
    scheduled_wait_time: Optional[int] = None  # Add this line


class PetriNet:
    def __init__(self):
        self.places = {}  #%
        self.transitions = {}  #%
        self.place_counter = 0  #%
        self.transition_counter = 0  #%
        self.max_tokens_at_planks_buffer = 0
        self.max_tokens_at_planks_buffer_idx = None
        self.token_history = []
        
    def list_tokens(self):
        """Print the tokens present in each place."""
        for p in self.places.values():  #%
            print(f"Place {p.id} has {p.tokens} tokens")
        
    def new_place(self, id: int = None, name: str = None) -> Place:
        """Create a new place."""
        if id is None:  #%
            id = self.place_counter  #%
            self.place_counter += 1  #%
        p = Place(id=id, name=name)
        self.places[id] = p  #%
        return p

    def get_state_vector(self) -> List[int]:
        state_vector = []
        for p in self.places.values():  #%
            state_vector.append(p.tokens)
            
    def new_transition(self, id: int = None, name: str = None, mu: float = 0, sigma_squared: float = 1) -> Transition:
        """Create a new transition."""
        if id is None:  #%
            id = self.transition_counter  #%
            self.transition_counter += 1  #%
        t = Transition(id=id, mu=mu, sigma_squared=sigma_squared, name=name)
        self.transitions[id] = t  #%
        return t

    def connect_input(self, weight: int, place_id: int, transition_id: int):
        if transition_id not in self.transitions or place_id not in self.places:  #%
            print("Error: Invalid place or transition ID")
            return

        t_curr = self.transitions[transition_id]  #%
        p_curr = self.places[place_id]  #%

        t_curr.in_arcs.append(Arc(weight=weight, place=p_curr))  #%

    def connect_output(self, weight: int, place_id: int, transition_id: int):
        if transition_id not in self.transitions or place_id not in self.places:  #%
            print("Error: Invalid place or transition ID")
            return

        t_curr = self.transitions[transition_id]  #%
        p_curr = self.places[place_id]  #%

        t_curr.out_arcs.append(Arc(weight=weight, place=p_curr))  #%

    def place_tokens(self, place_id: int, tokens: int):
        if place_id not in self.places:  #%
            print("Error: Invalid place ID")
            return
        self.places[place_id].tokens = tokens  #%

    def place_name(self, place_id: int, name: str):
        if place_id not in self.places:  #%
            print("Error: Invalid place ID")
            return
        self.places[place_id].name = name  #%

    def transition_name(self, transition_id: int, name: str):
        if transition_id not in self.transitions:  #%
            print("Error: Invalid transition ID")
            return
        self.transitions[transition_id].name = name  #%

    def visualize(self, filename="petri_net", step=0):
        dot = graphviz.Digraph(filename, engine='neato')
        dot.attr(overlap='false')
        dot.attr(fontsize='12')
        dot.attr(label=f"PetriNet Model - Step {step}")
        
        # Add places with token count and name (in blue)
        dot.attr('node', shape='circle', fixedsize='true', width='2.0', fontcolor='blue')   
        state_vector = []
        for p in self.places.values():  #%
            label = f'P{p.id}\nTokens: {p.tokens}'
            if p.name:
                label += f'\nName: {p.name}'
            dot.node(f'P{p.id}', label)
            state_vector.append(p.tokens)
        
        # Add transitions with name (in blue)
        dot.attr('node', shape='box', fontcolor='blue')
        for t in self.transitions.values():  #%
            label = f'T{t.id}'
            if t.name:
                label += f'\nName: {t.name}'
            dot.node(f'T{t.id}', label)
            
            # Add input arcs
            for arc in t.in_arcs:  #%
                dot.edge(f'P{arc.place.id}', f'T{t.id}', label=f'{arc.weight}')
                
            # Add output arcs
            for arc in t.out_arcs:  #%
                dot.edge(f'T{t.id}', f'P{arc.place.id}', label=f'{arc.weight}')
        
        filename_with_step = f"out/{filename}_step_{step}"
        dot.render(filename_with_step, format='png', cleanup=True)
        print(f"Graph rendered to {filename_with_step}.png with state vector: {state_vector}")

    def is_firable(self, transition: Transition) -> bool:
        """Check if a transition can be fired (all input arcs have enough tokens)."""
        for arc in transition.in_arcs:  #%
            if arc.place.tokens < arc.weight:
                return False
        return True

    def fire(self, transition: Transition):
        """Fire a transition (consume input tokens and produce output tokens)."""
        if not self.is_firable(transition):
            print(f"Transition T{transition.id} is not firable.")
            return False

        # Consume input tokens
        for arc in transition.in_arcs:  #%
            arc.place.tokens -= arc.weight

        # Produce output tokens
        for arc in transition.out_arcs:  #%
            arc.place.tokens += arc.weight

        print(f"Fired Transition T{transition.id}")
        return True

    def run_simulation(self, iterations=10, visualize=False):
        fire_matrix = []

        for i in range(iterations):
            print(f"\nIteration {i+1}:")
            self.list_tokens()
            if visualize:
                self.visualize(filename="petri_net_graph", step=i)

            fired_ids = [0] * len(self.transitions)  # Initialize with 0 (not fired)

            for t in self.transitions.values():
                if self.is_firable(t):
                    if t.scheduled_wait_time is None:
                        # Transition just became firable, schedule it
                        wait_time = int(np.random.normal(t.mu, np.sqrt(t.sigma_squared)))
                        wait_time = max(1, wait_time)  # Ensure positive wait time
                        t.scheduled_wait_time = wait_time
                        print(f"Transition T{t.id} scheduled to fire in {t.scheduled_wait_time} iterations.")
                    else:
                        # Transition is already scheduled, decrement wait time
                        t.scheduled_wait_time -= 1
                        if t.scheduled_wait_time <= 0:
                            # Fire the transition
                            self.fire(t)
                            fired_ids[t.id] = 1
                            t.scheduled_wait_time = None  # Reset wait time
                else:
                    # Transition is not firable, reset scheduled_wait_time
                    t.scheduled_wait_time = None
                    
            state_vectors = [place.tokens for place in self.places.values()]
            print(f'{state_vectors=}')
            self.token_history.append([i,state_vectors])
            if state_vectors[1] > self.max_tokens_at_planks_buffer:
                self.max_tokens_at_planks_buffer_idx = i
                self.max_tokens_at_planks_buffer = state_vectors[1]
            fire_matrix.append(fired_ids)

        # Print fire matrix after completing the simulation
        print("\nFire Matrix (Transitions Fired Each Iteration):")
        for idx, fired_transitions in enumerate(fire_matrix):
            print(f"Iteration {idx + 1}: Transitions Fired -> {fired_transitions}")
        print("#----------------------------------------------------------------#")
        print("Simulation completed.")
        print(f"MAX tokens at Planks Buffer: {self.max_tokens_at_planks_buffer} @ iteration {self.max_tokens_at_planks_buffer_idx}")
        [print(f"iteration {line[0]:>4} {np.array2string(np.array(line[1]), separator=' ', formatter={'all': lambda x: f'{x:>3}'})}") for line in self.token_history]




        
    def animate_petri_net(self, iterations=10, interval=1000):
        """
        Animate through Petri net simulation images with matplotlib.
        
        Args:
            iterations (int): Number of simulation steps to animate (default: 10)
            interval (int): Time between frames in milliseconds (default: 1000ms = 1s)
        """
        # Get list of PNG files sorted by step number
        image_files = sorted(
            glob.glob('out/petri_net_graph_step_*.png'),
            key=lambda x: int(x.split('step_')[1].split('.')[0])
        )
        
        if not image_files:
            print("No image files found in the out/ directory!")
            return
            
        # Limit the number of files to iterations
        image_files = image_files[:iterations]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(19, 19))
        plt.axis('off')
        
        # Initialize with first image
        img = mpimg.imread(image_files[0])
        im = ax.imshow(img)
        
        def update(frame):
            # Load and display the image for the current frame
            img = mpimg.imread(image_files[frame])
            im.set_array(img)
            return [im]
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, 
            update,
            frames=len(image_files),
            interval=interval,  # 1000ms = 1 second
            repeat=False,
            blit=True
        )
        
        plt.show()

import shutil

def clear_output_directory(directory="out"):
    """Deletes the specified directory and all its contents."""
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Deletes the directory and all its contents
        print(f"Deleted '{directory}' directory and its contents.")
    else:
        print(f"'{directory}' directory does not exist. No need to delete.")
        
def main():
    ## PARAM 
    iterations = 25
    spf = 0.5
    ## END PARAM
    pn = PetriNet()
    
    clear_output_directory()
    # Create 10 places
    for _ in range(10):
        pn.new_place()
    
    pn.place_name(place_id=0, name="Truck")
    pn.place_name(place_id=1, name="Wood_Buffer")
    pn.place_name(place_id=2, name="Tool")
    pn.place_name(place_id=3, name="ToolBusy")
    pn.place_name(place_id=4, name="P_Buffer_1")
    pn.place_name(place_id=5, name="P_Buffer_2")
    pn.place_name(place_id=6, name="Asm_1")
    pn.place_name(place_id=7, name="Asm_2")
    pn.place_name(place_id=8, name="Out Shelf 1")
    pn.place_name(place_id=9, name="Out Shelf 2")
    
    # Initialize tokens
    pn.place_tokens(place_id=0, tokens=1)
    pn.place_tokens(place_id=2, tokens=1)
    pn.place_tokens(place_id=6, tokens=1)
    pn.place_tokens(place_id=7, tokens=1)
    
    # Create 6 transitions
    # for _ in range(6):
    #     pn.new_transition()
        
    INPUT_PARAMETERS = False
    # SOFT_CODED
    if INPUT_PARAMETERS:
        print("#------------Enter the parameters for each transition------------#")
        for i, name in enumerate(["TREE_RATE","SAW_TIME","BORROW_TOOL","RETURN_TOOL","ASM_RATE_1","ASM_RATE_2"]):
            print("Enter param for transition ",name)
            mu = float(input("Enter mu: "))
            sigma_squared = float(input("Enter sigma_squared: "))
            print()
            pn.new_transition(id=i, name=name, mu=mu, sigma_squared=sigma_squared)
        print("#----------------------------------------------------------------#")
    
    # HARD_CODED
    print("#------------------parameters for each transition----------------#")
    parameters  = ("mu" , "sigma_squared")
    tree_rate   = (10    , 0.1)
    saw_time    = (0.1  , 0.1)
    borrow_tool = (100  , 0.1)
    return_tool = (1    , 1  )    
    asm_rate_1  = (5    , 0.1)
    asm_rate_2  = (5    , 0.1)
    print("#----------------------------------------------------------------#")
    
    pn.new_transition(id=0, name="tree_rate", mu=tree_rate[0], sigma_squared=tree_rate[1])
    pn.new_transition(id=1, name="saw_time", mu=saw_time[0], sigma_squared=saw_time[1])
    pn.new_transition(id=2, name="borrow_tool", mu=borrow_tool[0], sigma_squared=borrow_tool[1])
    pn.new_transition(id=3, name="return_tool", mu=return_tool[0], sigma_squared=return_tool[1])
    pn.new_transition(id=4, name="asm_rate_1", mu=asm_rate_1[0], sigma_squared=asm_rate_1[1])
    pn.new_transition(id=5, name="asm_rate_2", mu=asm_rate_2[0], sigma_squared=asm_rate_2[1])


    # Connect transitions with places
    # T0 TREE_RATE
    ## waiting for delivery of wood
    pn.connect_input(weight=1, place_id=0, transition_id=0)    # -1 token from P0
    pn.connect_output(weight=1, place_id=0, transition_id=0)   # +1 tokens to P0
    ## slice and into the buffer
    pn.connect_output(weight=10, place_id=1, transition_id=0)   # +4 tokens to P1
    
    # T1 SAW_TIME
    ## from the big wood bugger
    pn.connect_input(weight=2, place_id=1, transition_id=1)    # -1 token from P1
    ## tool avalability
    pn.connect_input(weight=1, place_id=2, transition_id=1)    # -1 token from P2
    pn.connect_output(weight=1, place_id=2, transition_id=1)   # +1 tokens to P2
    ## to planks buffer
    pn.connect_output(weight=2, place_id=4, transition_id=1)   # +2 tokens to P3
    pn.connect_output(weight=2, place_id=5, transition_id=1)   # +2 tokens to P4
    
    # T2 BORROW_TOOL 
    pn.connect_input(weight=1, place_id=2, transition_id=2)    # -1 token from P1
    pn.connect_output(weight=1, place_id=3, transition_id=2)   # +2 tokens to P4
    
    # T3 RETURN_TOOL 
    pn.connect_input(weight=1, place_id=3, transition_id=3)    # -1 token from P1
    pn.connect_output(weight=1, place_id=2, transition_id=3)   # +2 tokens to P4
    
    #* Parallel Shelving
    # T4 ASM_TIME_1
    ## worker avalability
    pn.connect_input(weight=1, place_id=6, transition_id=4)    # -1 token from P2
    pn.connect_output(weight=1, place_id=6, transition_id=4)   # +1 tokens to P2
    ## get planks from planks buffer
    pn.connect_input(weight=3, place_id=4, transition_id=4)    # -3 tokens from P3
    ## finsihed product
    pn.connect_output(weight=3, place_id=8, transition_id=4)   # +3 tokens to P7
    # T5 ASM_TIME_1
    ## worker avalability
    pn.connect_input(weight=1, place_id=7, transition_id=5)    # -1 token from P2
    pn.connect_output(weight=1, place_id=7, transition_id=5)   # +1 tokens to P2
    ## get planks from planks buffer
    pn.connect_input(weight=3, place_id=5, transition_id=5)    # -3 tokens from P3
    ## finsihed product
    pn.connect_output(weight=3, place_id=9, transition_id=5)   # +3 tokens to P7

    
    
    # Visualize initial state
    pn.visualize("petri_net_graph")
    
    # Run simulation without visualization in each step
    pn.run_simulation(iterations=iterations, visualize=True)  #%
    
    # Optionally, visualize final state
    pn.visualize("petri_net_graph_final")
    
    # If you need to animate, you can call the animate_petri_net method
    if not os.path.exists('out'):
        print("Please create 'out' directory and run the Petri net simulation first!")
    else:
        pn.animate_petri_net(iterations=iterations, interval=spf*1000)

if __name__ == "__main__":
    main()