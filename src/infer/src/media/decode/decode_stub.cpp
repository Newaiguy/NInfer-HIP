// HIP port stub: FFmpeg-based media decoding is not available in this environment.
// The mini_main text-only entry never invokes these; they exist only to satisfy the
// linker (frontend/processor.cpp references the interface for vision inputs).
// A real A-card vision port would implement these against a Windows decoder.
#include "media/decode/decode.h"

namespace ninfer::media::decode {

ImageInfo inspect_image(std::span<const std::uint8_t> /*bytes*/, const Policy& /*policy*/) {
    throw Error(ErrorKind::InvalidInput, "media decode: FFmpeg stub (vision input unsupported)");
}

VideoInfo inspect_video(std::span<const std::uint8_t> /*bytes*/, const Policy& /*policy*/,
                        double /*target_fps*/, int /*min_frames*/, int /*max_frames*/) {
    throw Error(ErrorKind::InvalidInput, "media decode: FFmpeg stub (vision input unsupported)");
}

Image decode_image(std::span<const std::uint8_t> /*bytes*/, const Policy& /*policy*/) {
    throw Error(ErrorKind::InvalidInput, "media decode: FFmpeg stub (vision input unsupported)");
}

Video decode_video(std::span<const std::uint8_t> /*bytes*/, const Policy& /*policy*/,
                   double /*target_fps*/, int /*min_frames*/, int /*max_frames*/) {
    throw Error(ErrorKind::InvalidInput, "media decode: FFmpeg stub (vision input unsupported)");
}

} // namespace ninfer::media::decode
