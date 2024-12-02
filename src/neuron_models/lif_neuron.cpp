#include "neuron_models/lif_neuron.hpp"

namespace snnlib
{
    
    void LIFNeuron::initialize(){};

    void LIFNeuron::setMembranePotential(double v, int index){
        x[index] = v;
    }
    
    void LIFNeuron::setMembranePotential(const std::vector<double>& v){
        assert(v.size() <= x.size());
        for(int i = 0; i < v.size(); i++){
            x[i] = v[i];
        }
    }
    
    double LIFNeuron::output_V(double* x, double* output_P, int t, int dt){
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

        double time_since_last_spike = t * dt - last_t;

        // Handle refractory period
        if (time_since_last_spike < t_ref) {
            return {0.0, 0.0};  // No variation in V or last_t
        }

        double dV = (-(V - V_rest) + I * R) / tau_m * dt; 

        if ((V + dV) >= V_th) {
            return {V_reset - V, t * dt - last_t};
        }

        return {dV, 0.0};
    }
    
} // namespace snnlib
