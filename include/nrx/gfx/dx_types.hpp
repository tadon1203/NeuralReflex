#pragma once

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmicrosoft-enum-forward-reference"
#endif
using D3D12_RESOURCE_STATES = enum D3D12_RESOURCE_STATES; // NOLINT(readability-identifier-naming)
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
