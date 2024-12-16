import PossionNeuron, LIFNeuron
import RestPotentialInitializer
import NormalWeightInitializer

build.neurons {
    "inputs": PossionNeuron [20];
    
    "reservoir": LIFNeuron [100] with init(RestPotentialInitializer);

    "outputs": LIFNeuron [16] with init(RestPotentialInitializer);
}

build.connections {
    "conn-inputs-outputs": CurrentBasedKernalSynapse with <"inputs" --> "reservoir">,
        init(NormalWeightInitializer);
    "conn-reservoir-outputs": CurrentBasedKernalSynapse with <"reservoir" --> "reservoir">,
        init(NormalWeightInitializer);
}


def_initializer {

}

def neuron {
    
}