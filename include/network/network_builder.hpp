
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
        NetworkBuilder(){
            _network = new snnlib::SNNNetwork();
        }
        
        snnlib::SNNNetwork* build_network(){
            _network->initialize();
            return _network;
        }

        int add_neuron(std::string neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron){
            int size = _network->neurons.size();
            _network->neurons[neuron_name] = neuron;
            return size;
        }
        
        int record_synapse(std::string synapse_name, std::shared_ptr<snnlib::AbstractSNNSynapse> synapse){
            int size = synapse_record.size();
            synapse_record[synapse_name] = synapse;
            return size;
        }

        int add_connection(std::string connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection){
            int size = _network->connections.size();
            _network->connections[connection_name] = connection;
            return size;
        }

        std::shared_ptr<snnlib::AbstractSNNNeuron> get_neuron(std::string neuron_name){
            return _network->neurons[neuron_name];
        }
        
        std::shared_ptr<snnlib::AbstractSNNConnection> get_connection(std::string connection_name){
            return _network->connections[connection_name];
        }

        std::shared_ptr<snnlib::AbstractSNNSynapse> get_synapse(std::string synapse_name){
            return synapse_record[synapse_name];
        }
        private:
            snnlib::SNNNetwork* _network;
            std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNSynapse>> synapse_record;
    };
}
#endif