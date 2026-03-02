#include "nrx/utils/dx_helper.hpp"

#include <sstream>

#include <Windows.h>

namespace nrx::utils {

auto DxHelper::getErrorString(HRESULT hr) -> std::string {
    LPSTR messageBuffer = nullptr;
    const DWORD size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, static_cast<DWORD>(hr), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&messageBuffer), 0, nullptr);

    if (size == 0 || messageBuffer == nullptr) {
        std::ostringstream stream;
        stream << "HRESULT 0x" << std::uppercase << std::hex << static_cast<unsigned int>(hr);
        return stream.str();
    }

    std::string message(messageBuffer, size);
    LocalFree(messageBuffer);

    while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) {
        message.pop_back();
    }

    return message;
}

} // namespace nrx::utils
