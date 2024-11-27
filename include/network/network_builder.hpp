
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
            _network->initialization();
            return _network;
        }


        int add_neuron(std::string neuron_name, snnlib::AbstractSNNNeuron* neuron){
            int size = _network->neurons.size();
            _network->neurons[neuron_name] = neuron;
            return size;
        }
        
        int record_synapse(std::string synapse_name, snnlib::AbstractSNNSynapse* synapse){
            int size = synapse_record.size();
            synapse_record[synapse_name] = synapse;
            return size;
        }
        int add_connection(std::string connection_name, snnlib::AbstractSNNConnection* connection){
            int size = _network->connections.size();
            _network->connections[connection_name] = connection;
            return size;
        }

        snnlib::AbstractSNNNeuron* get_neuron(std::string neuron_name){
            return _network->neurons[neuron_name];
        }
        
        snnlib::AbstractSNNConnection* get_connection(std::string connection_name){
            return _network->connections[connection_name];
        }

        int get_synapse(std::string conn){

        }
        private:
            snnlib::SNNNetwork* _network;
            std::unordered_map<std::string, snnlib::AbstractSNNSynapse*> synapse_record;
    };
}
#endif