#include "neuron_models/possion_neuron.hpp"
#include <random>

namespace snnlib
{
    void PossionNeuron::initialize(){};
    
    double PossionNeuron::output_V(int neuron_id, double* x, double* output_P, int t, double dt){
        // std::cout << "state_last_t(x)  = " << state_last_t(x)  << " t * dt = " << t * dt << std::endl;
        if(std::abs(state_last_t(x) - t * dt) < 1e-9){
            // std::cout << " - possion neuron (" << neuron_id << ")" << " output = 1" << std::endl;
            return 1.0;
        }
        return 0.0;
    }

    std::vector<double> PossionNeuron::neuron_dynamics(int neuron_id, double I, double* x, int t, double* P, double dt){
        DYN_SYSTEM_PARAMETER(freq);
        DYN_SYSTEM_STATE(last_t);
        DYN_SYSTEM_STATE(V);
        if((rand() / static_cast<double>(RAND_MAX)) < freq * dt){
            // std::cout << "possion neuron ( " << neuron_id << " ) emit spike." << std::endl;
            return {1.0 - V, t * dt - last_t};
        }
        return {0.0 - V, 0.0};
    }   
}