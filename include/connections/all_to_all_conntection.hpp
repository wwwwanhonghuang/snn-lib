#ifndef ALL_TO_ALL_CONNECTION_HPP
#define ALL_TO_ALL_CONNECTION_HPP
#include <memory>
#include "connections/connection.hpp"
namespace snnlib{
    struct AllToAllConnection: public AbstractSNNConnection
    {
        public:
            AllToAllConnection(std::shared_ptr<snnlib::AbstractSNNSynapse> synpases): 
                AbstractSNNConnection(synpases){
                    connected.assign(synapses->n_presynapse_neurons() * synapses->n_postsynapse_neurons(), true);
                }
    };
}
#endif