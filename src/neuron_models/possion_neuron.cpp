#include "neuron_models/possion_neuron.hpp"
#include <random>

namespace snnlib
{
    void PossionNeuron::initialize(){};
    
    double PossionNeuron::output_V(double* x, double* output_P, int t, double dt){
        if(state_last_t(x) == t * dt){
            // std::cout << "possion spiking." << std::endl;
            return 1.0;
        }
        return 0.0;
    }

    std::vector<double> PossionNeuron::neuron_dynamics(double I, double* x, double t, double* P, double dt){
        DYN_SYSTEM_PARAMETER(freq);
        DYN_SYSTEM_STATE(last_t);
        if((rand() / static_cast<double>(RAND_MAX)) < freq * dt){
            return {0.0, t * dt - last_t};
        }
        return {0.0, 0.0};
    }   
}