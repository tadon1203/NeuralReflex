#include "nrx/utils/dx_helper.hpp"

#include <sstream>

#include <Windows.h>

namespace nrx::utils {

namespace {

[[nodiscard]] auto hresultHex(std::int32_t hr) -> std::string {
    std::ostringstream stream;
    stream << "HRESULT 0x" << std::uppercase << std::hex << static_cast<unsigned int>(hr);
    return stream.str();
}

[[nodiscard]] auto wideToUtf8(const wchar_t* text, int textLength) -> std::string {
    const int utf8Size =
        WideCharToMultiByte(CP_UTF8, 0, text, textLength, nullptr, 0, nullptr, nullptr);
    if (utf8Size <= 0) {
        return {};
    }

    std::string utf8(static_cast<std::size_t>(utf8Size), '\0');
    const int converted = WideCharToMultiByte(CP_UTF8, 0, text, textLength, utf8.data(), utf8Size,
                                              nullptr, nullptr);
    if (converted <= 0) {
        return {};
    }
    return utf8;
}

[[nodiscard]] auto tryFormatMessage(DWORD messageId, DWORD languageId) -> std::string {
    LPWSTR messageBuffer = nullptr;
    const DWORD messageLength = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, messageId, languageId, reinterpret_cast<LPWSTR>(&messageBuffer), 0, nullptr);
    if (messageLength == 0 || messageBuffer == nullptr) {
        return {};
    }

    std::string message = wideToUtf8(messageBuffer, static_cast<int>(messageLength));
    LocalFree(messageBuffer);
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) {
        message.pop_back();
    }
    return message;
}

} // namespace

auto DxHelper::getErrorString(std::int32_t hr) -> std::string {
    const DWORD errorCode = static_cast<DWORD>(static_cast<unsigned long>(hr));

    // Prefer an English message for stable console readability across locales/code pages.
    auto message = tryFormatMessage(errorCode, MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US));
    if (message.empty()) {
        message = tryFormatMessage(errorCode, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT));
    }
    if (message.empty()) {
        return hresultHex(hr);
    }

    return hresultHex(hr) + " (" + message + ")";
}

} // namespace nrx::utils
