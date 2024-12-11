#include "network/network_builder.hpp"

namespace snnlib{
    static std::shared_ptr<snnlib::AbstractNeuronInitializer> neuron_rest_potential_initializer
        = std::make_shared<snnlib::RestPotentialInitializer>();

    static std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> connection_normal_weight_intializer
        = std::make_shared<snnlib::NormalWeightInitializer>();

    NetworkBuilder::NetworkBuilder(){
        _network = std::make_shared<snnlib::SNNNetwork>();
    }
    std::shared_ptr<SNNNetwork> NetworkBuilder::build_network(){
        for(auto& neuron_configuration_record : neuron_configuration_map){
            neuron_configuration_record.second->apply_initializer();
        }

        _network->initialize();
        return _network;
    }
   
    int NetworkBuilder::record_synapse(std::string synapse_name, std::shared_ptr<snnlib::AbstractSNNSynapse> synapse){
        int size = synapse_record.size();
        synapse_record[synapse_name] = synapse;
        return size;
    }

    std::shared_ptr<snnlib::AbstractSNNNeuron> NetworkBuilder::get_neuron(std::string neuron_name){
        return _network->neurons[neuron_name];
    }
    
    std::shared_ptr<snnlib::AbstractSNNConnection> NetworkBuilder::get_connection(std::string connection_name){
        return _network->connections[connection_name];
    }

    std::shared_ptr<snnlib::AbstractSNNSynapse> NetworkBuilder::get_synapse(std::string synapse_name){
        return synapse_record[synapse_name];
    }

    std::shared_ptr<snnlib::ConnectionConfiguration> 
        NetworkBuilder::register_connection(const std::string& connection_name, 
        std::shared_ptr<snnlib::AbstractSNNConnection> connection){
        if(connection_configuration_map.find(connection_name) != connection_configuration_map.end()){
            std::cerr << "Network Build Error: A connection with name " 
                << connection_name << " has already been recorded." << std::endl;
            return nullptr;
        }
        _network->connections[connection_name] = connection;
        return std::make_shared<snnlib::ConnectionConfiguration>(connection);
    }

   
        
}