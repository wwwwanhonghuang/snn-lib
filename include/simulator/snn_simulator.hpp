#ifndef SNN_SIMULATOR_HPP
#define SNN_SIMULATOR_HPP
#include <iostream>
#include "network/network.hpp"

#include "recorder/neuron_recorder.hpp"

namespace snnlib {
    struct SNNSimulator
    {
        private:
            snnlib::NeuronRecorder _neuron_recorder;
        public:
            SNNSimulator(){

            }
            void simulate(std::shared_ptr<snnlib::SNNNetwork> network, int time_steps, double dt){
                for(int t = 0; t < time_steps; t++){
                    std::cout << "* In time step " << t << std::endl;

                    network->evolve_states(t, dt);
                    network->global_update();
                }
            }
    };
}
#endif