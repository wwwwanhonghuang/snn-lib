#ifndef SNN_SIMULATOR_HPP
#define SNN_SIMULATOR_HPP
#include <iostream>
#include "network/network.hpp"
namespace snnlib{
    struct SNNSimulator
    {
        /* data */

        public:
            void simulate(snnlib::SNNNetwork* network, int time_steps){
                for(int t = 0; t < time_steps; t++){
                    std::cout << "* In time step " << t << std::endl;
                    network->evolve_states();
                    network->global_update();
                }
            }
    };
}
#endif