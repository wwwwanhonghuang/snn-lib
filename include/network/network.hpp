#ifndef NETWORK_HPP
#define NETWORK_HPP
#include <unordered_map>
#include <string>
#include <iostream>
#include <cassert>

#include "neuron_models/neuron.hpp"
#include "connections/connection.hpp"
#include "recorder/simulation_state_recorder.hpp"

namespace snnlib{
    struct SNNNetwork{
        // name -> neuron mapping
        std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNNeuron>> neurons;
        std::unordered_map<std::string, std::shared_ptr<snnlib::AbstractSNNConnection>> connections;
        std::unordered_map<std::shared_ptr<snnlib::AbstractSNNNeuron>, std::string> neuron_name_map;
        std::unordered_map<std::shared_ptr<snnlib::AbstractSNNConnection>, std::string> connection_name_map;
        
        // name -> id mapping
        std::unordered_map<std::string, int> neuron_id_map;
        std::unordered_map<std::string, int> connection_id_map;
        std::vector<std::vector<std::shared_ptr<snnlib::AbstractSNNConnection>>> connection_matrix;

        bool is_neuron_connected(const std::string& presynapse_neuron_name, const std::string& postsynapse_neuron_name);

        
        void initialize();

        void global_update();

        void evolve_states(int t, double dt);
    };
}
#endif