#ifndef SNN_SIMULATOR_HPP
#define SNN_SIMULATOR_HPP
#include <iostream>
#include "network/network.hpp"

#include "recorder/recorder.hpp"


namespace snnlib {
    struct SNNSimulator
    {
        private:
        public:
            SNNSimulator(){

            }
            void simulate(std::shared_ptr<snnlib::SNNNetwork> network, int time_steps, double dt, std::shared_ptr<snnlib::RecorderFacade> record_facade = nullptr){
                for(int t = 0; t < time_steps; t++){
                    std::cout << "* In time step " << t << std::endl;
                    
                    network->evolve_states(t, dt, record_facade);
                    network->global_update();
                }
                finish_processing();
            }

            void finish_processing(){

            }
    };
}
#endif