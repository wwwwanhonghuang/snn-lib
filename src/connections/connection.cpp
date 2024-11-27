#include "connections/connection.hpp"

namespace snnlib
{
    void AbstractSNNConnection::initialize_weights(std::shared_ptr<snnlib::AbstractSNNConnectionWeightInitializer> initializer){
        initializer->initialize(std::shared_ptr<snnlib::AbstractSNNConnection>(this));   
    }
} // namespace snnlib
