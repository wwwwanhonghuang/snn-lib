#ifndef NEURON_HPP
#define NEURON_HPP
#include <vector>
#include <cstring>
#include <stdexcept>

#include "interfaces/function.hpp"
namespace snnlib{
    struct AbstractSNNNeuron
    {
        public:
            int n_neurons;
            std::vector<double> P;

            NeuronDynamicsModel neuron_dynamics_model;

            virtual void initialize() = 0;
            virtual void setMembranePotential(double v, int index) = 0;
            virtual void setMembranePotential(const std::vector<double>& v) = 0;

            void forward_states_to_buffer(const std::vector<double>& I, double t, double* P, double dt){
                for(int i = 0; i < n_neurons; i++){
                    std::vector<double> dx = neuron_dynamics_model(I[i], &x[i * n_states], t, P, dt);
                    for(int state_id = 0; state_id < n_states; state_id++){
                        double x_i = x[i * n_states + state_id] + dx[state_id];
                        x_buffer[i * n_states + state_id] = x_i;
                    }
                }
            }

            void update_states_from_buffer(){
                std::memcpy(x.data(), x_buffer.data(), x.size() * sizeof(double));
            }

            AbstractSNNNeuron(int n_neurons, int n_states)
                            :n_neurons(n_neurons), n_states(n_states),
                x(n_neurons * n_states, 0.0),
                x_buffer(n_neurons * n_states, 0.0) {
                if (n_neurons <= 0 || n_states <= 0) {
                    throw std::invalid_argument("n_neurons and n_states must be positive.");
                }
            }

        protected:
            int n_states;
            int n_parameters;
            std::vector<double> x;
            std::vector<double> x_buffer;
            

        private:
            
            
            void _evolve_state(const std::vector<double>& I, double t, double* P, double dt){
                for(int i = 0; i < n_neurons; i++){
                    std::vector<double> dx = neuron_dynamics_model(I[i], &x[i * n_states], t, P, dt);
                    for(int state_id = 0; state_id < n_states; state_id++){
                        double x_i = x[i * n_states + state_id] + dx[state_id];
                        x[i * n_states + state_id] = x_i;
                    }
                }
            }
    };
}
#endif