#ifndef CONNECTION_HPP
#define CONNECTION_HPP
#include "synapse_models/synapse.hpp"
namespace snnlib{
    struct AbstractSNNConnection
    {
        public:
            snnlib::AbstractSNNSynapse* synapses;
            std::vector<double> weights;
            AbstractSNNConnection(snnlib::AbstractSNNSynapse& synapses): 
                synapses(&synapses){
                weights.assign(synapses.n_presynapse_neurons() * synapses.n_postsynapse_neurons(), 0);
            };

            void forward_states_to_buffer(double weight, const std::vector<double>& S, double t, double* P, double dt){
                
            }
            void update_states_from_buffer(){

            }
    };
}
#endif