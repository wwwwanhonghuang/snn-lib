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

void build_neurons(std::shared_ptr<snnlib::NetworkBuilder> network_builder){
    network_builder->build_neuron<snnlib::PossionNeuron>("inputs", 1, 80000);
    network_builder->build_neuron<snnlib::LIFNeuron>("reservoir", 1)->add_initializer("rest_potential_initializer");
    network_builder->build_neuron<snnlib::LIFNeuron>("outputs", 1)->add_initializer("rest_potential_initializer");
}

void build_connections(std::shared_ptr<snnlib::NetworkBuilder> network_builder) {
    std::shared_ptr<snnlib::IdenticalWeightInitializer> initializer =
        std::make_shared<snnlib::IdenticalWeightInitializer>(0.5);
    
    // rise, decay
    network_builder->build_synapse<snnlib::CurrentBasedKernalSynapse>(
        "syn_input_reservoir", "inputs", "reservoir", "double_exponential", 1e-2, 1e-2, 0.0, 0.0)
        ->build_connection<snnlib::AllToAllConnection>("conn-input-reservoir", network_builder)
        ->add_initializer(initializer);

    network_builder->build_synapse<snnlib::CurrentBasedKernalSynapse>(
        "syn_reservoir_output", "reservoir", "outputs", "double_exponential",  1e-2,  1e-2, 0.0, 0.0)
        ->build_connection<snnlib::AllToAllConnection>("conn-reservoir-output", network_builder)
        ->add_initializer(initializer);
}

void run_simulation(std::shared_ptr<snnlib::SNNNetwork> network, 
                        int time_steps, double dt, 
                        std::shared_ptr<snnlib::RecorderFacade> recorder_facade = nullptr){
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
    std::shared_ptr<snnlib::NetworkBuilder> network_builder = 
        std::make_shared<snnlib::NetworkBuilder>();
    
    build_neurons(network_builder);
    build_connections(network_builder);

    std::shared_ptr<snnlib::SNNNetwork> network = network_builder->build_network();


    std::shared_ptr<snnlib::NeuronRecorder> neuron_recorder =
        std::make_shared<snnlib::NeuronRecorder>();
    std::shared_ptr<snnlib::ConnectionRecorder> connection_recorder =
        std::make_shared<snnlib::ConnectionRecorder>();

    // Build Recorder
    snnlib::ConnectionRecordCallback weight_recorder = [](const std::string& connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection, int t, int dt) -> void{
        if(t == 0)
            snnlib::WeightRecorder::record_connection_weights_to_file(std::string("data/logs/") + connection_name + std::string(".weights"), connection);
    };

    snnlib::ConnectionRecordCallback response_recorder = [connection_recorder](const std::string& connection_name, std::shared_ptr<snnlib::AbstractSNNConnection> connection, int t, int dt) -> void{
        connection_recorder->record_synapse_response_to_file(std::string("data/logs/") + connection_name + std::string(".r"), connection, t);
    };

    snnlib::NeuroRecordCallback membrane_potential_recorder = [neuron_recorder](const std::string& neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron, int t, int dt) -> void{
        neuron_recorder->record_membrane_potential_to_file(
            std::string("data/logs/")  + neuron_name
            + std::string(".v"), neuron, t
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