#include "network/network_builder.hpp"

namespace snnlib
{
    SynapseConfiguration::SynapseConfiguration(std::shared_ptr<snnlib::AbstractSNNSynapse> synapse){
        this->_synapse = synapse;
    };   
} // namespace snnlib
