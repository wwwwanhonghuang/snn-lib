#ifndef NEURON_INITIALIZER_HPP
#define NEURON_INITIALIZER_HPP
#include "neuron_models/neuron.hpp"
namespace snnlib
{
    struct AbstractNeuronMembranePotentialInitializer
    {
    };

    struct RestPotentialInitializer : snnlib::AbstractNeuronMembranePotentialInitializer
    {
        void initialize(snnlib::AbstractSNNNeuron& neuron){
            neuron.setMembranePotential(-65.0);
        }
    };
    
}
#endif