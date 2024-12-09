#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <yaml-cpp/yaml.h>

#include "neuron_models/neuron.hpp"
#include "neuron_models/lif_neuron.hpp"
#include "neuron_models/possion_neuron.hpp"
#include "neuron_models/initializer.hpp"

#include "network/network.hpp"
#include "network/network_builder.hpp"

#include "synapse_models/synapse.hpp"

#include "simulator/snn_simulator.hpp"

#include "network/initializer/weight_initializers.hpp"

#include "recorder/recorder.hpp"
#include "recorder/connection_recorder.hpp"
#include "recorder/neuron_recorder.hpp"

#include "connections/all_to_all_conntection.hpp"

void build_neurons(snnlib::NetworkBuilder& network_builder){
    network_builder.build_neuron<snnlib::PossionNeuron>("inputs", 1, 80000);
    network_builder.build_neuron<snnlib::LIFNeuron>("reservoir", 1)->add_initializer("rest_potential_initializer");
    network_builder.build_neuron<snnlib::LIFNeuron>("outputs", 1)->add_initializer("rest_potential_initializer");
}

void build_synapses(snnlib::NetworkBuilder& network_builder) {
    // rise, decay
    network_builder.build_synapse<snnlib::CurrentBasedKernalSynapse>(
        "syn_input_reservoir", "inputs", "reservoir", "double_exponential", 1e-2, 1e-2, 0, 0);
    network_builder.build_synapse<snnlib::CurrentBasedKernalSynapse>(
        "syn_reservoir_output", "reservoir", "outputs", "double_exponential",  1e-2,  1e-2, 0, 0);
}

void establish_connections(snnlib::NetworkBuilder& network_builder){
    std::shared_ptr<snnlib::IdenticalWeightInitializer> initializer =
        std::make_shared<snnlib::IdenticalWeightInitializer>(0.5);
    network_builder.build_connection<snnlib::AllToAllConnection>(
        "conn-input-reservoir", "syn_input_reservoir")->add_initializer(initializer);
    network_builder.build_connection<snnlib::AllToAllConnection>(
        "conn-reservoir-output", "syn_reservoir_output")->add_initializer(initializer);
}

void run_simulation(std::shared_ptr<snnlib::SNNNetwork> network, int time_steps, double dt,  std::shared_ptr<snnlib::RecorderFacade> recorder_facade = nullptr){
    snnlib::SNNSimulator simulator;
    simulator.simulate(network, time_steps, dt, recorder_facade);
}

int main(){
    YAML::Node config;
    try {
        config = YAML::LoadFile("config.yaml");
    } catch (const std::exception& e) {
        std::cerr << "Error loading config.yaml: " << e.what() << std::endl;
        return -1;
    }

    // Read Configuration
    int time_steps = config["snn-main"]["time-steps"].as<int>();
    double dt = config["snn-main"]["dt"].as<double>();

    std::cout << "time_steps = " << time_steps << std::endl;
    std::cout << "dt = " << dt << std::endl;

    // Build Network
    snnlib::NetworkBuilder network_builder;
    build_neurons(network_builder);
    build_synapses(network_builder);
    establish_connections(network_builder);

    std::shared_ptr<snnlib::SNNNetwork> network = network_builder.build_network();

    // Build Recorder
    snnlib::ConnectionRecordCallback weight_recorder = [](const std::string& connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection, int t, int dt) -> void{
        if(t == 0)
            snnlib::WeightRecorder::record_connection_weights_to_file(std::string("data/logs/") + connection_name + std::string(".weights"), connection);
    };

    snnlib::NeuroRecordCallback membrane_potential_recorder = [](const std::string& neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron, int t, int dt) -> void{
        snnlib::NeuronRecorder::record_membrane_potential_to_file(
            std::string("data/logs/")  + neuron_name
            + std::string("_t_") + std::to_string(t)
            + std::string(".v"), neuron
        );
    };

    // Build Recorder
    std::shared_ptr<snnlib::RecorderFacade> recorder_facade = 
                            std::make_shared<snnlib::RecorderFacade>();
    recorder_facade->add_connection_record_item("conn-input-reservoir", weight_recorder);
    recorder_facade->add_connection_record_item("conn-reservoir-output", weight_recorder);

    recorder_facade->add_neuron_record_item("outputs", membrane_potential_recorder);
    recorder_facade->add_neuron_record_item("reservoir", membrane_potential_recorder);

    // Run Simulation
    run_simulation(network, time_steps, dt, recorder_facade);
}