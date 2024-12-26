#ifndef ALL_TO_ALL_CONNECTION_HPP
#define ALL_TO_ALL_CONNECTION_HPP
#include <memory>
#include "connections/connection.hpp"
namespace snnlib{
   

    struct AllToAllConnection: public AbstractSNNConnection
    {

        public:
            AllToAllConnection(std::shared_ptr<snnlib::AbstractSNNSynapse> synapses): 
                AbstractSNNConnection(synapses){
                    connected.assign(synapses->n_presynapse_neurons() * synapses->n_postsynapse_neurons(), true);
            }

            std::vector<double> process_pre_neuron_spike_trains(const std::vector<double>& S, int t, double* P, double dt) override{
                std::vector<double> processed_S(S.size());
                int index = 0;
                for(size_t i = 0; i < S.size(); i++){
                    processed_S[i] = weights[i] * S[i];
                }
                return processed_S;
            }

    };
}
#endif