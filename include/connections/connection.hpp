#ifndef CONNECTION_HPP
#define CONNECTION_HPP
#include <memory>
#include "synapse_models/synapse.hpp"
#include "network/initializer/initializer.hpp"

namespace snnlib{
    struct AbstractSNNConnectionInitializer;
}
namespace snnlib{
    

    struct AbstractSNNConnection
    {
        public:
            std::shared_ptr<snnlib::AbstractSNNSynapse> synapses;
            std::vector<bool> connected;
            std::vector<double> P;
            std::vector<double> weights;


            // currently connections are not a dynamical systems.
            // std::vector<double> states;
            // std::vector<double> state_buffer;
            // int n_states = 1;

            AbstractSNNConnection(std::shared_ptr<snnlib::AbstractSNNSynapse> synapses): 
                synapses(synapses), 
                weights(synapses->n_presynapse_neurons() * synapses->n_postsynapse_neurons(), 0){
            };

            void initialize_weights(std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> initializer);
            void forward_states_to_buffer(const std::vector<double>& S, int t, double* P, double dt);
            void update_states_from_buffer();
            virtual std::vector<double> process_pre_neuron_spike_trains(const std::vector<double>& S, int t, double* P, double dt){
                return S;
            };

            virtual void initialize(){}
    };
    
}
#endif