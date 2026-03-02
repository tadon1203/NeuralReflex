#include <exception>

#include "nrx/core/application.hpp"
#include "nrx/utils/logger.hpp"

int main() {
    nrx::utils::Logger::init();
    NRX_INFO("NeuralReflex Starting...");

    try {
        nrx::core::Application app;
        return app.run();

    } catch (const std::exception& e) {
        NRX_CRITICAL("Fatal Error: {}", e.what());
        return -1;
    }
}
