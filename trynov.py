from dataclasses import dataclass
from typing import Optional
import graphviz

@dataclass
class Place:
    id: int
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
    in_arcs: Optional[Arc] = None
    out_arcs: Optional[Arc] = None
    next: Optional['Transition'] = None

class PetriNet:
    def __init__(self):
        self.places = None
        self.transitions = None
        
    def new_place(self) -> Place:
        p = Place(id=0)
        if self.places is None:
            self.places = p
            return p
        
        tmp = self.places
        p.id = 1
        while tmp.next is not None:
            tmp = tmp.next
            p.id += 1
        tmp.next = p
        return p
    
    def new_transition(self) -> Transition:
        t = Transition(id=0)
        if self.transitions is None:
            self.transitions = t
            return t
        
        tmp = self.transitions
        t.id = 1
        while tmp.next is not None:
            tmp = tmp.next
            t.id += 1
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
    
        
    def visualize(self, filename="petri_net"):
        dot = graphviz.Digraph(filename, engine='neato')
        dot.attr(overlap='false')
        dot.attr(fontsize='12')
        dot.attr(label="PetriNet Model")
        
        # Add places
        p = self.places
        dot.attr('node', shape='circle', fixedsize='true', width='0.9')
        while p:
            label = f'P{p.id}\nTokens: {p.tokens}'
            dot.node(f'P{p.id}', label)
            p = p.next
        
        # Add transitions
        t = self.transitions
        dot.attr('node', shape='box')
        while t:
            dot.node(f'T{t.id}', f'T{t.id}')
            
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
        
        dot.render(filename, format='png', cleanup=True)
        print(f"Graph BRUHHHHHHHH to {filename}.png")

def main():
    pn = PetriNet()
    
    # Create 7 places
    for _ in range(11):
        pn.new_place()
        
    # Place initial tokens
    pn.place_tokens(place_id=0, tokens=3)
    pn.place_tokens(place_id=1, tokens=1)
    
    # Create 4 transitions
    for _ in range(6):
        pn.new_transition()
        
    # Connect transitions with places
    
    # T0
    pn.connect_output(weight=4, place_id=1, transition_id=0) # +4 token
    pn.connect_input(weight=1, place_id=0, transition_id=0) # -1 token
    pn.connect_output(weight=1, place_id=0, transition_id=0) # +1 token
    
    # T1
    pn.connect_input(weight=1, place_id=1, transition_id=1) # -1 token
    pn.connect_output(weight=1, place_id=2, transition_id=1) # +1 token
    pn.connect_input(weight=1, place_id=2, transition_id=1) # -1 token
    pn.connect_output(weight=2, place_id=3, transition_id=1) # +1 token
    pn.connect_output(weight=2, place_id=4, transition_id=1) # +1 token
    
    # T2
    pn.connect_input(weight=3, place_id=3, transition_id=2) # -1 token
    pn.connect_output(weight=3, place_id=7, transition_id=2) # +1 token
    
    # T3
    pn.connect_input(weight=3, place_id=4, transition_id=3) # -1 token
    pn.connect_output(weight=3, place_id=8, transition_id=3) # +1 token
    
    # T4
    pn.connect_input(weight=1, place_id=5, transition_id=4) # -1 token
    pn.connect_output(weight=1, place_id=5, transition_id=4) # +1 token
    pn.connect_input(weight=3, place_id=7, transition_id=4) # -1 token
    pn.connect_output(weight=1, place_id=9, transition_id=4) # +1 token

    # T5
    pn.connect_input(weight=1, place_id=6, transition_id=5) # -1 token
    pn.connect_output(weight=1, place_id=6, transition_id=5) # +1 token
    pn.connect_input(weight=3, place_id=8, transition_id=5) # -1 token
    pn.connect_output(weight=1, place_id=10, transition_id=5) # +1 token
    
    # Visualize
    pn.visualize("petri_net_graph")

if __name__ == "__main__":
    main()