#ifndef LIF_NEURON_HPP
#define LIF_NEURON_HPP
#include <vector>
#include <cassert>
#include "neuron.hpp"

#include "macros.def"
namespace snnlib{
    struct LIFNeuron: public AbstractSNNNeuron
    {
        DEF_NEURON_PARAM(0, V_rest);
        DEF_NEURON_PARAM(1, V_th);
        DEF_NEURON_PARAM(2, V_reset);
        DEF_NEURON_PARAM(3, tau_m);
        DEF_NEURON_PARAM(4, R);
        DEF_NEURON_PARAM(5, t_ref);

        DEF_NEURON_STATE(0, V);
        DEF_NEURON_STATE(1, last_t);


        LIFNeuron(int n_neurons, double V_rest = -65.0, double V_th = -50.0, double V_reset = -70.0, 
            double tau_m = 20.0, double t_ref = 0.0025, double R = 1.0): AbstractSNNNeuron(n_neurons, 2)
        {
            neuron_dynamics_model = &LIFNeuron::neuron_dynamics;
            this->n_neurons = n_neurons;
            P.assign({V_rest, V_th, V_reset, tau_m, R, t_ref});
        }

        virtual void initialize(){

        };

        virtual void setMembranePotential(double v, int index){
            x[index] = v;
        }
        
        virtual void setMembranePotential(const std::vector<double>& v){
            assert(v.size() <= x.size());
            for(int i = 0; i < v.size(); i++){
                x[i] = v[i];
            }
        }

        static std::vector<double> neuron_dynamics(double I, double* x, double t, double* P, double dt){
            NEURON_STATE(V);
            NEURON_STATE(last_t);

            NEURON_PARAMETER(V_rest);
            NEURON_PARAMETER(V_th);
            NEURON_PARAMETER(V_reset);
            NEURON_PARAMETER(tau_m);
            NEURON_PARAMETER(R);
            NEURON_PARAMETER(t_ref);

            double time_since_last_spike = t * dt - last_t;

            // Handle refractory period
            if (time_since_last_spike < t_ref) {
                // No changes during refractory period
                return {0.0, 0.0};  // No variation in V or last_t
            }

            // Compute membrane potential dynamics
            double dV = (-(V - V_rest) + I * R) / tau_m * dt; 

            // Check for a spike
            if ((V + dV) >= V_th) {
                // Spike: reset voltage and update last spike time
                return {V_reset - V, t * dt - last_t};
            }

            // No spike: return variation in V, no change to last_t
            return {dV, 0.0};
        }
        
    };
}
#endif