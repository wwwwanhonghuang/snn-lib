#ifndef POSSION_NEURON_HPP
#define POSSION_NEURON_HPP
#include "macros.def"
#include "neuron_models/neuron.hpp"

namespace snnlib{
    struct PossionNeuron: public AbstractSNNNeuron
    {
        DEF_DYN_SYSTEM_PARAM(0, freq);

        DEF_DYN_SYSTEM_STATE(1, last_t);

        PossionNeuron(int n_neurons, int frequency, double t_ref = 0.0): AbstractSNNNeuron(n_neurons, 2)
        {
            neuron_dynamics_model = &PossionNeuron::neuron_dynamics;
            this->n_neurons = n_neurons;
            P.assign({(double)frequency});
        }

        virtual void initialize();
        
        virtual double output_V(double* x, double* output_P, int t, double dt);

        static std::vector<double> neuron_dynamics(double I, double* x, double t, double* P, double dt);
    };

}
#endif