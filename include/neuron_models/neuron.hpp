#ifndef NEURON_HPP
#define NEURON_HPP
#include <vector>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include "interfaces/function.hpp"
#include "macros.def"
#include <cassert>
namespace snnlib{
    struct AbstractSNNNeuron
    {
        public:
            DEF_DYN_SYSTEM_STATE(0, V)

            int n_neurons;
            std::vector<double> P;
            std::unordered_map<std::string, int> parameter_map;

            NeuronDynamicsModel neuron_dynamics_model;

            virtual void initialize() = 0;

            inline void setMembranePotential(int index, double v){
                x[index * n_states + OFFSET_STATE_V] = v;
            }
            
            void setMembranePotential(const std::vector<double>& mV){
                assert(mV.size() <= x.size());
                for(size_t i = 0; i < mV.size(); i++){
                    setMembranePotential(i, mV[i]);
                }
            }


            void setMembranePotential(double mV){
                for(int i = 0; i < n_neurons; i++){
                    setMembranePotential(i, mV);
                }
            }
            
            virtual double output_V(int neuron_id, double* x, double* output_P, int t, double dt) = 0;
            inline void forward_states_to_buffer(int neuron_index, double I, int t, double* P, double dt);
            void forward_states_to_buffer(const std::vector<double>& I, int t, double* P, double dt);

            void update_states_from_buffer();

            AbstractSNNNeuron(int n_neurons, int n_states);
            int get_n_states();
            int n_states;
            int n_parameters;
            std::vector<double> x;
            std::vector<double> x_buffer;

            double get_parameters(const std::string& parameter_name){
                if(parameter_map.find(parameter_name) == parameter_map.end()){
                    throw std::invalid_argument(std::string("cannot find neuron parameter with name: ") + parameter_name);
                }
                return P[parameter_map[parameter_name]];
            };
        
        protected:

        private:
            void _evolve_state(const std::vector<double>& I, int t, double* P, double dt);
    };
}
#endif