from halp.directed_hypergraph import DirectedHypergraph
import sympy as sp
from string import ascii_letters

class Species:

    size: int
    value: sp.Symbol

    def __init__(
            self,
            size,
    ):
        self.size = size
        self.value = sp.symbols("B"+str(self.size))

    def __repr__(self):
        return 'B'+str(self.size)
    
    def __hash__(self):
        return hash(self.size)

    def __eq__(self, other):
        return self.size==other

    def __gt__(self, other):
        return self.value>other

    def __ge__(self, other):
        return self.value>=other

    def __lt__(self, other):
        return self.value<other

    def __le__(self, other):
        return self.value<=other

    def __len__(self):
        return self.size
    
    def __add__(self, other):
        return self.value+other

    def __radd__(self, other):
        return self.value+other
    
    def __sub__(self, other):
        return self.value-other
    
    def __mul__(self, other):
        return self.value*other
    
    def __rmul__(self, other):
        return self.value*other
    
    def __truediv__(self, other):
        return self.value/other

    def __pow__(self, power, modulo=None):
        return self.value**power




class Reaction:

    reactants: dict[Species: int]
    product: Species
    forward_rate: sp.Symbol
    reverse_rate: sp.Symbol

    def __init__(
            self,
            reactants,
            product,
            forward_rate=None,
            reverse_rate=None,
                 ):
        self.reactants = reactants
        self.product = product
        self.forward_rate = forward_rate
        self.reverse_rate = reverse_rate
        # if forward_rate is None or reverse_rate is None:
        #     self.forward_rate = sp.Symbol(self.current_sample_rate)
        #     Reaction.update_rates()
        #     self.reverse_rate = sp.Symbol(self.current_sample_rate)
        #     Reaction.update_rates()

    
    def __repr__(self):
        rep = []
        for reactant, coef in self.reactants.items():
            rep.append(f"{coef}{reactant}")
        rep = " + ".join(rep)
        rep += f" <{self.reverse_rate}--{self.forward_rate}> "
        rep += str(self.product)
        return rep

    def __eq__(self, other):
        return self.reactants==other.reactants and self.product is other.product

    def __contains__(self, other):
        if other in self.reactants.keys() or other is self.product:
            return True

    def get_diffeq_terms(self, spec):

        if spec is self.product:
            pos = self.forward_rate
            for reactant, coef in self.reactants.items():
                pos *= reactant.value**coef
            neg = self.reverse_rate*spec
            return pos - neg
        
        pos = self.reverse_rate*self.reactants[spec]*self.product.value
        neg = self.forward_rate*self.reactants[spec]
        for reactant, coef in self.reactants.items():
            neg *= reactant**coef
        
        return pos-neg




class SpeciesHypergraph:

    sizes: list
    species: list
    graph: DirectedHypergraph
    current_sample_rate: str


    def __init__(self, sizes):

        self.sizes = sorted(sizes)
        self.species = [Species(size) for size in self.sizes]
        self.species_dict = {size: self.species[i] for i, size in enumerate(self.sizes)}
        self.current_sample_rate = "a1"


    def __repr__(self):
        rep = ""
        for id in sorted(self.graph.get_hyperedge_id_set(), key=lambda x: int(x.strip(ascii_letters))):
            rep += str(id) + " " + str(self.graph.get_hyperedge_attributes(id)) + "\n"
        return rep


    def update_rates(self):
        let, num = self.current_sample_rate
        num = int(num)
        if num == 9:
            let = chr(ord(let)+1)
            num = 1
        else:
            num += 1
        self.current_sample_rate = let+str(num)


    def create_graph(
            self,
            include_multiterm_rxns=True,
            include_offladder_rxns=True
            ):

        graph = DirectedHypergraph()
        graph.add_nodes(self.species)

        for spec in self.species:
            parts = list(self._partition(spec.size))[:-1]
            parts = [i for i in parts if all([j in self.sizes for j in i])]
            parts = [[self.species_dict[j] for j in i] for i in parts]

            for part in parts:
                reactants = {r: part.count(r) for r in list(set(part))}
                frate = sp.Symbol(self.current_sample_rate)
                self.update_rates()
                rrate = sp.Symbol(self.current_sample_rate)
                self.update_rates()
                rxn = Reaction(reactants, spec, forward_rate=frate, reverse_rate=rrate)

                if graph.has_hyperedge(list(reactants.keys()), [spec]):
                    existing_rxns = graph.get_hyperedge_attribute(graph.get_hyperedge_id(reactants, [spec]), "rxn")
                    graph.add_hyperedge(list(reactants.keys()), [spec], {'rxn': existing_rxns+[rxn]})
                else:
                    graph.add_hyperedge(list(reactants.keys()), [spec], {'rxn': [rxn]})

        self.graph = graph

        if not include_multiterm_rxns:
            self._remove_multiterm_reactions()
        
        if not include_offladder_rxns:
            self._remove_multiterm_reactions()
            self._remove_offladder_reactions()
        


    def _partition(self, n):
        a,k,y=[0 for i in range(n)],1,n-1
        while k!=0:
            x,k=a[k-1]+1,k-1
            while 2*x<=y:
                a[k],y,k=x,y-x,k+1
            l=k+1
            while x<=y:
                a[k],a[l]=x,y
                yield a[:k+2]
                x,y=x+1,y-1
            a[k],y=x+y,x+y-1
            yield a[:k+1]


    def remove_species(self, size):
        self.species.pop(self.sizes.index(size))
        self.sizes.remove(size)
        self.species_dict.pop(size)
        self.create_graph()


    def remove_reaction(self, reactants, product):
        reaction = Reaction({self.species_dict[size]: coef for size, coef in reactants.items()}, self.species_dict[product])
        remove_ids = []
        for id in self.graph.hyperedge_id_iterator():
            rxns = self.graph.get_hyperedge_attribute(id, 'rxn')
            if reaction in rxns:
                if len(rxns) == 1:
                    remove_ids.append(id)
                else:
                    rxns.remove(reaction)
                    self.graph.add_hyperedge(list(reactants.keys()), [product], {'rxn': rxns})
        self.graph.remove_hyperedges(remove_ids)

    
    def _remove_multiterm_reactions(self):
        remove_rxns = []
        for id in self.graph.hyperedge_id_iterator():
            if len(self.graph.get_hyperedge_tail(id)) > 1:
                for rxn in self.graph.get_hyperedge_attribute(id, 'rxn'):
                    remove_rxns.append(rxn)
        for rxn in remove_rxns:
            self.remove_reaction(rxn.reactants, rxn.product)

    
    def _remove_offladder_reactions(self):
        self._remove_multiterm_reactions()
        remove_rxns = []
        for id in self.graph.hyperedge_id_iterator():
            index = self.species.index(self.graph.get_hyperedge_tail(id)[0])
            if self.graph.get_hyperedge_head(id)[0] is not self.species[index+1]:
                for rxn in self.graph.get_hyperedge_attribute(id, 'rxn'):
                    remove_rxns.append(rxn)
        for rxn in remove_rxns:
            self.remove_reaction(rxn.reactants, rxn.product)

    def get_diffeq(self, size):
        expr = 0
        for rxns in self.get_all_rxns(size):
            for rxn in rxns:
                expr += rxn.get_diffeq_terms(self.species_dict[size])
        return expr


    def get_all_diffeqs(self):
        diffeqs = {}
        for size in self.sizes:
            diffeqs[size] = self.get_diffeq(size)
        return diffeqs


    def get_all_rxns(self, size) -> list[Reaction]:
        rxns = []
        for hypid in self.graph.hyperedge_id_iterator():
            if self.species_dict[size] in self.graph.get_hyperedge_head(hypid) + self.graph.get_hyperedge_tail(hypid):
                rxns.append(self.graph.get_hyperedge_attribute(hypid, 'rxn'))
        return rxns

if __name__ == "__main__":
    # B1 = Species(1)
    # B12 = Species(12)
    # B24 = Species(24)
    # a1, a2 = sp.symbols("a1 a2")

    # rxn1 = Reaction(
    #     {B1: 12},
    #     B12,
    #     a1, a2
    # )

    # print(rxn1.get_diffeq_terms())

    graph = SpeciesHypergraph([1, 12, 24])
    graph.create_graph()
    print(graph)
    graph.create_graph(include_offladder_rxns=False)
    print(graph)

    graph.get_diffeq(1)
