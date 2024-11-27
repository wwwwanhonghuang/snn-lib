#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>
#include <span>
#include <memory> 
#include "neuron_models/neuron.hpp"
#include "neuron_models/lif_neuron.hpp"
#include "synapse_models/synapse.hpp"
#include "connections/connection.hpp"
#include "connections/all_to_all_conntection.hpp"

#include "interfaces/function.hpp"
#include "simulator/snn_simulator.hpp"
#include "network/network.hpp"
#include "network/network_builder.hpp"

// Function to build neurons
std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNNeuron>> build_neurons() {
    std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNNeuron>> neurons;
    neurons["input"] = std::make_shared<snnlib::LIFNeuron>(100);
    neurons["output"] = std::make_shared<snnlib::LIFNeuron>(200);
    return neurons;
}

// Function to build synapses
std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNSynapse>> 
build_synapses(std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNNeuron>>& neurons) {
    std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNSynapse>> synapses;
    synapses["syn-input-output"] = std::make_shared<snnlib::ConvolutionCurrentBasedSynapse>(
        neurons["input"].get(), neurons["output"].get());
    return synapses;
}

std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNConnection>> 
build_connections(std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNSynapse>>& synapse){
    std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNConnection>> connections;
    connections["connection-inputs-outputs"] = std::make_shared<snnlib::AllToAllConnection>(
        *synapse["syn-input-output"]
    );
    return connections;
}

snnlib::SNNNetwork* build_network(std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNNeuron>> neurons, 
    std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNConnection>> connections){
        snnlib::NetworkBuilder network_builder;
        for(auto& neuron_record_item: neurons){
            network_builder.add_neuron(neuron_record_item.first, neuron_record_item.second.get());
        }
        for(auto& connection_record_item: connections){
            network_builder.add_connection(connection_record_item.first, connection_record_item.second.get());
        }
        return network_builder.build_network();
}

void simulation_main(snnlib::SNNNetwork* motion_capture_network, int time_steps){
    snnlib::SNNSimulator simulator;
    simulator.simulate(motion_capture_network, time_steps);
}

int main(){
    std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNNeuron>> neurons = build_neurons();
    std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNSynapse>> synapses = build_synapses(neurons);
    std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNConnection>> connections = build_connections(synapses);
    snnlib::SNNNetwork* network = build_network(neurons, connections);

    int time_step = 2000;
    simulation_main(network, time_step);
}