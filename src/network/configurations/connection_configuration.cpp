#include "network/network_builder.hpp"

namespace snnlib
{
    std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> 
        ConnectionConfiguration::get_predefined_initializer(const std::string& initializer_name){
        if (initializer_name == "connection_normal_initializer") {
            return connection_normal_weight_intializer;
        } else {
            std::cerr << "Error: unrecognized connection initializer " << initializer_name << std::endl;
            assert(false);
            return nullptr;
        }
    }

    std::shared_ptr<snnlib::ConnectionConfiguration> 
        ConnectionConfiguration::add_initializer(const std::string& initializer_name) {
        return add_initializer(get_predefined_initializer(initializer_name));
    }

    std::shared_ptr<snnlib::ConnectionConfiguration> 
        ConnectionConfiguration::add_initializer(std::shared_ptr<snnlib::AbstractSNNConnectionInitializer> 
                initializer) {
        if(initializer)
            _initializers.push_back(initializer);
        return shared_from_this();
    }

    void ConnectionConfiguration::apply_initializer() {
        for(auto& initializer : _initializers){
            initializer->initialize(_connection);
        }
    }

    ConnectionConfiguration::ConnectionConfiguration(std::shared_ptr<snnlib::AbstractSNNConnection> connection){
        this->_connection = connection;
    }

} // namespace snnlib
