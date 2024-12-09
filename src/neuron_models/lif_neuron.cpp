#include "neuron_models/lif_neuron.hpp"
#include <cmath>
namespace snnlib
{
    
    void LIFNeuron::initialize(){};
    
    double LIFNeuron::output_V(double* x, double* output_P, int t, double dt){
        if(state_last_t(x) == t * dt){
            return 1.0;
        }else{
            return 0.0;
        }
    }

    std::vector<double> LIFNeuron::neuron_dynamics(double I, double* x, double t, double* P, double dt){
        DYN_SYSTEM_STATE(V);
        DYN_SYSTEM_STATE(last_t);

        DYN_SYSTEM_PARAMETER(V_rest);
        DYN_SYSTEM_PARAMETER(V_th);
        DYN_SYSTEM_PARAMETER(V_reset);
        DYN_SYSTEM_PARAMETER(tau_m);
        DYN_SYSTEM_PARAMETER(R);
        DYN_SYSTEM_PARAMETER(t_ref);

        double time_since_last_spike = (last_t == -1) ? INFINITY : t * dt - last_t;

        if (time_since_last_spike < t_ref) {
            return {0.0, 0.0};
        }

        double dV = (-(V - V_rest) + I * R) / tau_m * dt; 
        if ((V + dV) >= V_th) {
            return {V_reset - V, t * dt - last_t};
        }

        return {dV, 0.0};
    }
    
} // namespace snnlib
