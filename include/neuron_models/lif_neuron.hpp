#ifndef LIF_NEURON_HPP
#define LIF_NEURON_HPP
#include <vector>
#include <cassert>
#include "neuron.hpp"

#include "macros.def"

namespace snnlib{
    struct LIFNeuron: public AbstractSNNNeuron
    {
        DEF_DYN_SYSTEM_PARAM(0, V_rest);
        DEF_DYN_SYSTEM_PARAM(1, V_th);
        DEF_DYN_SYSTEM_PARAM(2, V_reset);
        DEF_DYN_SYSTEM_PARAM(3, tau_m);
        DEF_DYN_SYSTEM_PARAM(4, R);
        DEF_DYN_SYSTEM_PARAM(5, t_ref);

        DEF_DYN_SYSTEM_STATE(1, last_t);

        LIFNeuron(int n_neurons, double V_rest = -65.0, double V_th = -50.0, double V_reset = -70.0, 
            double tau_m = 20.0, double t_ref = 0.0025, double R = 10.0): AbstractSNNNeuron(n_neurons, 2)
        {
            neuron_dynamics_model = &LIFNeuron::neuron_dynamics;
            this->n_neurons = n_neurons;
            for(int i = 0; i < n_neurons; i++){
                x[i * n_states + OFFSET_STATE_last_t] = -1;
            }
            P.assign({V_rest, V_th, V_reset, tau_m, R, t_ref});
        }

        virtual void initialize();
        
        virtual double output_V(double* x, double* output_P, int t, double dt);

        static std::vector<double> neuron_dynamics(double I, double* x, double t, double* P, double dt);
    };

}
#endif