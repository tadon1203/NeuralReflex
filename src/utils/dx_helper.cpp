#include "nrx/utils/dx_helper.hpp"

#include <sstream>

#include <Windows.h>

namespace nrx::utils {

namespace {

[[nodiscard]] auto hresultHex(HRESULT hr) -> std::string {
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

auto DxHelper::getErrorString(HRESULT hr) -> std::string {
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

auto DxHelper::createBuffer(ID3D12Device* d12Device, std::uint64_t sizeBytes,
                            D3D12_RESOURCE_FLAGS flags, D3D12_HEAP_TYPE heapType)
    -> DxResult<ID3D12Resource> {
    if (d12Device == nullptr || sizeBytes == 0) {
        return std::unexpected(E_INVALIDARG);
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
    D3D12_HEAP_PROPERTIES heapProperties{};
    heapProperties.Type = heapType;

    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Width = sizeBytes;
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.Format = DXGI_FORMAT_UNKNOWN;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    description.Flags = flags;
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)

    const auto initialState = (heapType == D3D12_HEAP_TYPE_READBACK)
                                  ? D3D12_RESOURCE_STATE_COPY_DEST
                                  : D3D12_RESOURCE_STATE_COMMON;

    winrt::com_ptr<ID3D12Resource> resource;
    const HRESULT createHr =
        d12Device->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &description,
                                           initialState, nullptr, IID_PPV_ARGS(resource.put()));
    if (FAILED(createHr)) {
        return std::unexpected(createHr);
    }

    return resource;
}

auto DxHelper::createUavBuffer(ID3D12Device* d12Device, std::uint64_t sizeBytes)
    -> DxResult<ID3D12Resource> {
    return createBuffer(d12Device, sizeBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        D3D12_HEAP_TYPE_DEFAULT);
}

auto DxHelper::createReadbackBuffer(ID3D12Device* d12Device, std::uint64_t sizeBytes)
    -> DxResult<ID3D12Resource> {
    return createBuffer(d12Device, sizeBytes, D3D12_RESOURCE_FLAG_NONE, D3D12_HEAP_TYPE_READBACK);
}

} // namespace nrx::utils
