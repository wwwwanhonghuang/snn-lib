
#ifndef NETWORK_BUILDER_HPP
#define NETWORK_BUILDER_HPP
#include <vector>
#include <unordered_map>
#include "network/network.hpp"
#include "neuron_models/neuron.hpp"
#include "connections/connection.hpp"

namespace snnlib{
    struct NetworkBuilder
    {        
        NetworkBuilder();
        snnlib::SNNNetwork* build_network();
        int add_neuron(std::string neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron);
        int record_synapse(std::string synapse_name, std::shared_ptr<snnlib::AbstractSNNSynapse> synapse);
        int add_connection(std::string connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection);

        std::shared_ptr<snnlib::AbstractSNNNeuron> get_neuron(std::string neuron_name);
        std::shared_ptr<snnlib::AbstractSNNConnection> get_connection(std::string connection_name);
        std::shared_ptr<snnlib::AbstractSNNSynapse> get_synapse(std::string synapse_name);
        private:
            snnlib::SNNNetwork* _network;
            std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNSynapse>> synapse_record;
    };
}
#endif