// Windows media decoding for the HIP port.
//
// Upstream uses FFmpeg. This port has no FFmpeg headers or libraries, and vision input still has to
// work, so image decoding goes through Windows Imaging Component (WIC) instead: it ships with the OS,
// understands PNG/JPEG/GIF/BMP/TIFF, and needs no extra binaries next to the executable.
//
// Video input stays unsupported until an FFmpeg build is available; the throw is explicit so callers
// get a clear message instead of a decoder that silently returns nothing.

#include "media/decode/decode.h"

#include <windows.h>
#include <wincodec.h>

#include <cstdio>
#include <cstring>
#include <vector>

#pragma comment(lib, "windowscodecs.lib")
// COM apartment setup and the HGLOBAL-backed stream live in ole32.
#pragma comment(lib, "ole32.lib")

namespace ninfer::media::decode {
namespace {

// WIC wants COM up on the decoding thread; the engine calls into here from request threads.
void ensure_com() {
    static thread_local bool ready = false;
    if (ready) { return; }
    const HRESULT status = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (SUCCEEDED(status) || status == RPC_E_CHANGED_MODE) { ready = true; }
}

std::string hr_text(HRESULT status) {
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "0x%08lx", static_cast<unsigned long>(status));
    return buffer;
}

struct ComPtr {
    IUnknown* object = nullptr;

    ComPtr() = default;
    ComPtr(const ComPtr&) = delete;
    ComPtr& operator=(const ComPtr&) = delete;
    ComPtr(ComPtr&& other) noexcept : object(other.object) { other.object = nullptr; }
    ComPtr& operator=(ComPtr&& other) noexcept {
        if (this != &other) {
            if (object != nullptr) { object->Release(); }
            object       = other.object;
            other.object = nullptr;
        }
        return *this;
    }
    ~ComPtr() {
        if (object != nullptr) { object->Release(); }
    }

    template <typename T>
    T** put() {
        return reinterpret_cast<T**>(&object);
    }
    template <typename T>
    T* get() const {
        return reinterpret_cast<T*>(object);
    }
};

// Wraps the encoded bytes in an IStream backed by an HGLOBAL; WIC reads straight from it.
ComPtr stream_over(std::span<const std::uint8_t> bytes) {
    HGLOBAL block = GlobalAlloc(GMEM_MOVEABLE, bytes.empty() ? 1 : bytes.size());
    if (block == nullptr) {
        throw Error(ErrorKind::InvalidInput, "media decode: GlobalAlloc failed");
    }
    void* target = GlobalLock(block);
    if (target == nullptr) {
        GlobalFree(block);
        throw Error(ErrorKind::InvalidInput, "media decode: GlobalLock failed");
    }
    if (!bytes.empty()) { std::memcpy(target, bytes.data(), bytes.size()); }
    GlobalUnlock(block);

    ComPtr stream;
    if (FAILED(CreateStreamOnHGlobal(block, TRUE, stream.put<IStream>()))) {
        GlobalFree(block);
        throw Error(ErrorKind::InvalidInput, "media decode: CreateStreamOnHGlobal failed");
    }
    return stream;
}

ComPtr wic_factory() {
    ComPtr factory;
    const HRESULT status = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                            IID_PPV_ARGS(factory.put<IWICImagingFactory>()));
    if (FAILED(status)) {
        throw Error(ErrorKind::InvalidInput, "media decode: WIC factory unavailable " + hr_text(status));
    }
    return factory;
}

// Opens the first frame of whatever image the bytes describe.
ComPtr open_frame(std::span<const std::uint8_t> bytes, ComPtr* factory) {
    ensure_com();
    if (bytes.size() > (256ULL << 20)) {
        throw Error(ErrorKind::InvalidInput, "media decode: input exceeds the byte budget");
    }

    *factory       = wic_factory();
    ComPtr stream  = stream_over(bytes);
    ComPtr decoder;
    HRESULT status = factory->get<IWICImagingFactory>()->CreateDecoderFromStream(
        stream.get<IStream>(), nullptr, WICDecodeMetadataCacheOnDemand,
        decoder.put<IWICBitmapDecoder>());
    if (FAILED(status)) {
        throw Error(ErrorKind::InvalidInput,
                    "media decode: unsupported or corrupt image (WIC " + hr_text(status) + ")");
    }

    ComPtr frame;
    status = decoder.get<IWICBitmapDecoder>()->GetFrame(0, frame.put<IWICBitmapFrameDecode>());
    if (FAILED(status)) {
        throw Error(ErrorKind::InvalidInput, "media decode: image frame unavailable " + hr_text(status));
    }
    return frame;
}

void check_budget(UINT width, UINT height, const Policy& policy) {
    const std::uint64_t pixels =
        static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    if (pixels > policy.max_decoded_pixels) {
        throw Error(ErrorKind::BudgetExceeded, "media decode: image exceeds the pixel budget");
    }
    if (policy.checkpoint) { policy.checkpoint(); }
}

}  // namespace

ImageInfo inspect_image(std::span<const std::uint8_t> bytes, const Policy& policy) {
    ComPtr factory;
    ComPtr frame = open_frame(bytes, &factory);
    UINT width   = 0;
    UINT height  = 0;
    if (FAILED(frame.get<IWICBitmapFrameDecode>()->GetSize(&width, &height))) {
        throw Error(ErrorKind::InvalidInput, "media decode: image size unavailable");
    }
    check_budget(width, height, policy);
    ImageInfo info;
    info.width  = static_cast<int>(width);
    info.height = static_cast<int>(height);
    return info;
}

VideoInfo inspect_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    throw Error(ErrorKind::InvalidInput,
                "media decode: video input needs the FFmpeg build, which this port does not ship");
}

Image decode_image(std::span<const std::uint8_t> bytes, const Policy& policy) {
    ComPtr factory;
    ComPtr frame = open_frame(bytes, &factory);

    UINT width  = 0;
    UINT height = 0;
    if (FAILED(frame.get<IWICBitmapFrameDecode>()->GetSize(&width, &height))) {
        throw Error(ErrorKind::InvalidInput, "media decode: image size unavailable");
    }
    check_budget(width, height, policy);
    if (width == 0 || height == 0) {
        throw Error(ErrorKind::InvalidInput, "media decode: empty image");
    }

    // BGRA is the one format every WIC decoder can produce, so conversion never depends on a
    // codec-specific path being installed.
    ComPtr converter;
    HRESULT status = factory.get<IWICImagingFactory>()->CreateFormatConverter(
        converter.put<IWICFormatConverter>());
    if (FAILED(status)) {
        throw Error(ErrorKind::InvalidInput, "media decode: converter unavailable " + hr_text(status));
    }
    status = converter.get<IWICFormatConverter>()->Initialize(
        frame.get<IWICBitmapFrameDecode>(), GUID_WICPixelFormat32bppBGRA, WICBitmapDitherTypeNone,
        nullptr, 0.0, WICBitmapPaletteTypeCustom);
    if (FAILED(status)) {
        throw Error(ErrorKind::InvalidInput, "media decode: pixel conversion failed " + hr_text(status));
    }

    const UINT stride = width * 4U;
    std::vector<std::uint8_t> bgra(static_cast<std::size_t>(stride) * height);
    status = converter.get<IWICFormatConverter>()->CopyPixels(
        nullptr, stride, static_cast<UINT>(bgra.size()), bgra.data());
    if (FAILED(status)) {
        throw Error(ErrorKind::InvalidInput, "media decode: pixel copy failed " + hr_text(status));
    }

    // Composite over white so a transparent PNG matches what the FFmpeg path upstream produces.
    Image image;
    image.width  = static_cast<int>(width);
    image.height = static_cast<int>(height);
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    image.rgb.resize(pixels * 3U);
    for (std::size_t index = 0; index < pixels; ++index) {
        const std::uint8_t blue  = bgra[index * 4U + 0U];
        const std::uint8_t green = bgra[index * 4U + 1U];
        const std::uint8_t red   = bgra[index * 4U + 2U];
        const std::uint8_t alpha = bgra[index * 4U + 3U];
        const auto blend         = [alpha](std::uint8_t channel) {
            const std::uint32_t value =
                static_cast<std::uint32_t>(channel) * alpha + 255U * (255U - alpha);
            return static_cast<std::uint8_t>((value + 127U) / 255U);
        };
        image.rgb[index * 3U + 0U] = blend(red);
        image.rgb[index * 3U + 1U] = blend(green);
        image.rgb[index * 3U + 2U] = blend(blue);
    }
    return image;
}

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    throw Error(ErrorKind::InvalidInput,
                "media decode: video input needs the FFmpeg build, which this port does not ship");
}

}  // namespace ninfer::media::decode
