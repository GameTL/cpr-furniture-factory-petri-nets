from dataclasses import dataclass
from typing import Optional
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
    name: Optional['str'] = None
    tokens: int = 0
    next: Optional['Place'] = None

@dataclass
class Arc:
    weight: int
    place: Place
    next: Optional['Arc'] = None

@dataclass
class Transition:
    id: int
    name: Optional['str'] = None
    in_arcs: Optional[Arc] = None
    out_arcs: Optional[Arc] = None
    next: Optional['Transition'] = None
    mu: float = 0.0  # Mean for Gaussian distribution
    sigma_squared: float = 1.0  # Variance for Gaussian distribution (sigma^2)

class PetriNet:
    def __init__(self):
        self.places = None
        self.transitions = None
        
    def list_tokens(self):
        """Print the tokens present in each place."""
        p = self.places
        while p:
            print(f"Place {p.id} has {p.tokens} tokens")
            p = p.next
        
    def new_place(self, id: int = None, name: str = None) -> Place:
        """ 
        Create a new place.
        if places is None, place id = 0. Otherwise, traverse the linked list to the end and create a new place.
        - If `id` is provided, use that as the place ID, otherwise auto-generate it.
        - `name` can be provided for visualization purposes.
        """
        p = Place(id=id if id is not None else 0, name=name)
        if self.places is None:
            # Set the initial ID if none is provided
            if id is None:
                p.id = 0
            self.places = p
            return p
        
        tmp = self.places
        if id is None:
            p.id = 1
            while tmp.next is not None:
                tmp = tmp.next
                p.id += 1
        else:
            p.id = id
        tmp.next = p
        return p
    
    def new_transition(self, id: int = None, name: str = None , mu : float = 0, sigma_squared: float = 1) -> Transition:
        """
        Create a new transition.
        - If `id` is provided, use that as the transition ID, otherwise auto-generate it.
        - `name` can be provided for visualization purposes.
        """
        t = Transition(id=id if id is not None else 0, mu=mu, sigma_squared=sigma_squared, name=name)
        if self.transitions is None:
            # Set the initial ID if none is provided
            if id is None:
                t.id = 0
            self.transitions = t
            return t
        
        tmp = self.transitions
        if id is None:
            t.id = 1
            while tmp.next is not None:
                tmp = tmp.next
                t.id += 1
        else:
            t.id = id
        tmp.next = t
        return t

    
    def connect_input(self, weight: int, place_id: int, transition_id: int):
        t_curr = self.transitions
        p_curr = self.places
        
        while t_curr and t_curr.id != transition_id:
            t_curr = t_curr.next
        while p_curr and p_curr.id != place_id:
            p_curr = p_curr.next
            
        if not t_curr or not p_curr:
            print("Error: Invalid place or transition ID")
            return
            
        arc = Arc(weight=weight, place=p_curr)
        arc.next = t_curr.in_arcs
        t_curr.in_arcs = arc
        
    def connect_output(self, weight: int, place_id: int, transition_id: int):
        t_curr = self.transitions
        p_curr = self.places
        
        while t_curr and t_curr.id != transition_id:
            t_curr = t_curr.next
        while p_curr and p_curr.id != place_id:
            p_curr = p_curr.next
            
        if not t_curr or not p_curr:
            print("Error: Invalid place or transition ID")
            return
            
        arc = Arc(weight=weight, place=p_curr)
        arc.next = t_curr.out_arcs
        t_curr.out_arcs = arc
        
    def place_tokens(self, place_id: int, tokens: int):
        p_curr = self.places
        while p_curr and p_curr.id != place_id:
            p_curr = p_curr.next
            
        if not p_curr:
            print("Error: Invalid place ID")
            return
            
        p_curr.tokens = tokens
    def place_name(self, place_id: int, name: str):
        p_curr = self.places
        while p_curr and p_curr.id != place_id:
            p_curr = p_curr.next
            
        if not p_curr:
            print("Error: Invalid place ID")
            return
            
        p_curr.name = name
    def transition_name(self, transition_id: int, name: str):
        t_curr = self.transitions
        while t_curr and t_curr.id != transition_id:
            t_curr = t_curr.next
            
        if not t_curr:
            print("Error: Invalid transition ID")
            return
            
        t_curr.name = name
    
        
    def visualize(self, filename="petri_net", step=0):
            dot = graphviz.Digraph(filename, engine='neato')
            dot.attr(overlap='false')
            dot.attr(fontsize='12')
            dot.attr(label=f"PetriNet Model - Step {step}")
            
            # Add places with token count and name (in blue)
            p = self.places
            dot.attr('node', shape='circle', fixedsize='true', width='0.9', fontcolor='blue')
            # dot.attr('node', shape='circle', fixedsize='true', width='1.9', fontcolor='blue')
            state_vector = []
            while p:
                label = f'P{p.id}\nTokens: {p.tokens}'
                if p.name:
                    label += f'\nName: {p.name}'
                dot.node(f'P{p.id}', label)
                state_vector.append(p.tokens)
                p = p.next
            
            # Add transitions with name (in blue)
            t = self.transitions
            dot.attr('node', shape='box', fontcolor='blue')
            while t:
                label = f'T{t.id}'
                if t.name:
                    label += f'\nName: {t.name}'
                dot.node(f'T{t.id}', label)
                
                # Add input arcs
                arc = t.in_arcs
                while arc:
                    dot.edge(f'P{arc.place.id}', f'T{t.id}', label=f'{arc.weight}')
                    arc = arc.next
                    
                # Add output arcs
                arc = t.out_arcs
                while arc:
                    dot.edge(f'T{t.id}', f'P{arc.place.id}', label=f'{arc.weight}')
                    arc = arc.next
                    
                t = t.next
            
            filename_with_step = f"out/{filename}_step_{step}"
            dot.render(filename_with_step, format='png', cleanup=True)
            print(f"Graph rendered to {filename_with_step}.png with state vector: {state_vector}")


    def is_firable(self, transition: Transition) -> bool:
        """Check if a transition can be fired (all input arcs have enough tokens)."""
        arc = transition.in_arcs
        while arc:
            if arc.place.tokens < arc.weight:
                return False  # Not enough tokens to fire
            arc = arc.next
        return True

    def fire(self, transition: Transition):
        """Fire a transition (consume input tokens and produce output tokens)."""
        if not self.is_firable(transition):
            print(f"Transition T{transition.id} is not firable.")
            return False

        # Consume input tokens
        arc = transition.in_arcs
        while arc:
            if arc.place.tokens >= arc.weight:  # Double-check this condition
                arc.place.tokens -= arc.weight
            else:
                print(f"Error: Not enough tokens in Place {arc.place.id}")
                return False
            arc = arc.next

        # Produce output tokens
        arc = transition.out_arcs
        while arc:
            arc.place.tokens += arc.weight
            arc = arc.next

        print(f"Fired Transition T{transition.id}")
        return True

    def run_simulation(self, iterations=10):
        for i in range(iterations):
            print(f"\nIteration {i+1}:")
            self.list_tokens()
            self.visualize(filename="petri_net_graph", step=i)

            # Collect all firable transitions
            t = self.transitions
            firable_transitions = []
            gaussians = []  # Collect Gaussian parameters for firable transitions
            while t:
                if self.is_firable(t):
                    firable_transitions.append(t)
                    gaussians.append((t.mu, np.sqrt(t.sigma_squared)))  # Store (mean, std_dev)
                t = t.next

            if not firable_transitions:
                print("No transitions can be fired. Stopping simulation.")
                break

            # Monte Carlo step using Gaussian distribution
            # Generate a random index based on the Gaussian distribution for each transition
            probabilities = []
            for idx, (mu, sigma) in enumerate(gaussians):
                # Generate a random value from a normal distribution and compute probability
                random_val = np.random.normal(mu, sigma)
                probabilities.append((random_val, idx))
            
            # Sort based on the generated random values (higher preference for higher values)
            probabilities.sort(reverse=True, key=lambda x: x[0])
            selected_transition = firable_transitions[probabilities[0][1]]  # Select the transition with highest value

            self.fire(selected_transition)
        print("Simulation completed.")
        
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
def main():
    ## PARAM 
    
    iterations = 10
    
    ## END PARAM
    pn = PetriNet()
    
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
    
    # all loop must have 1 token for the valid firing
    pn.place_tokens(place_id=0, tokens=1)
    pn.place_tokens(place_id=2, tokens=1)
    pn.place_tokens(place_id=6, tokens=1)
    pn.place_tokens(place_id=7, tokens=1)
    
    # Create 6 transitions
    for _ in range(6):
        pn.new_transition()
    # Connect transitions with places
    # T0 TREE_RATE
    ## waiting for delivery of wood
    pn.connect_input(weight=1, place_id=0, transition_id=0)    # -1 token from P0
    pn.connect_output(weight=1, place_id=0, transition_id=0)   # +1 tokens to P0
    ## slice and into the buffer
    pn.connect_output(weight=4, place_id=1, transition_id=0)   # +4 tokens to P1
    
    # T1 SAW_TIME
    ## from the big wood bugger
    pn.connect_input(weight=1, place_id=1, transition_id=1)    # -1 token from P1
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
    pn.connect_output(weight=1, place_id=8, transition_id=4)   # +3 tokens to P7
    
    # T5 ASM_TIME_1
    ## worker avalability
    pn.connect_input(weight=1, place_id=7, transition_id=5)    # -1 token from P2
    pn.connect_output(weight=1, place_id=7, transition_id=5)   # +1 tokens to P2
    ## get planks from planks buffer
    pn.connect_input(weight=3, place_id=5, transition_id=5)    # -3 tokens from P3
    ## finsihed product
    pn.connect_output(weight=1, place_id=9, transition_id=5)   # +3 tokens to P7

    
    # Visualize initial state
    pn.visualize("petri_net_graph")
    
    # Run simulation
    pn.run_simulation(iterations=iterations)
    
    # Ensure output directory exists
    if not os.path.exists('out'):
        print("Please create 'out' directory and run the Petri net simulation first!")
    else:
        pn.animate_petri_net(iterations=iterations)

if __name__ == "__main__":
    main()