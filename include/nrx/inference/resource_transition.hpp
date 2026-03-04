#pragma once

#include "nrx/gfx/dx_types.hpp"

namespace nrx::inference {

struct ResourceTransition {
    D3D12_RESOURCE_STATES fromState;
    D3D12_RESOURCE_STATES toState;
};

} // namespace nrx::inference
