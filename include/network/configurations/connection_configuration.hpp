#ifndef CONNECTION_CONFIGURATION_HPP
#define CONNECTION_CONFIGURATION_HPP
#include <memory>
#include <cassert>
#include "network/initializer/initializer.hpp"
#include "network/initializer/weight_initializers.hpp"

namespace snnlib{
class ConnectionConfiguration {
        public:
            std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> 
                get_predefined_initializer(const std::string& initializer_name){
                if (initializer_name == "connection_normal_initializer") {
                    return connection_normal_weight_intializer;
                } else {
                    std::cerr << "Error: unrecognized connection initializer " << initializer_name << std::endl;
                    assert(false);
                }
            }
            // Add an initializer and return the current instance for chaining
            std::shared_ptr<snnlib::ConnectionConfiguration> add_initializer(const std::string& initializer_name) {
                return add_initializer(get_predefined_initializer(initializer_name));
            }

            std::shared_ptr<snnlib::ConnectionConfiguration> add_initializer(std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> 
                initializer) {
                if(initializer)
                    _initializers.push_back(initializer);
                return std::make_shared<snnlib::ConnectionConfiguration>(this);
            }

            // Apply configurations to finalize
            void apply_initializer() {
                for(auto& initializer : _initializers){
                    initializer->initialize(_connection);
                }
            }
            ConnectionConfiguration(std::shared_ptr<snnlib::AbstractSNNConnection> connection){
                this->_connection = connection;
            }
        private:
            std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> connection_normal_weight_intializer =
            std::make_shared<snnlib::NormalWeightInitializer>();
          
        
            std::vector<std::shared_ptr<snnlib::AbstractSNNConnectionInitializer>> _initializers;
            std::shared_ptr<snnlib::AbstractSNNConnection> _connection;
};
}
#endif