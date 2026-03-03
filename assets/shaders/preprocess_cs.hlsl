Texture2D<float4> inputTexture : register(t0);
RWStructuredBuffer<float> outputTensor : register(u0);

SamplerState linearClampSampler : register(s0);

cbuffer PreprocessConstants : register(b0) {
    uint srcWidth;
    uint srcHeight;
    uint dstWidth;
    uint dstHeight;
    float scale;
    float padX;
    float padY;
    float inv255;
};

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID) {
    if (dispatchThreadId.x >= dstWidth || dispatchThreadId.y >= dstHeight) {
        return;
    }

    const float2 dstPixel = float2(dispatchThreadId.xy);
    const float2 scaledSize = float2(srcWidth, srcHeight) * scale;
    const float2 letterboxMin = float2(padX, padY);
    const float2 letterboxMax = letterboxMin + scaledSize;

    float3 rgb = float3(0.0F, 0.0F, 0.0F);

    const bool insideContent = dstPixel.x >= letterboxMin.x && dstPixel.x < letterboxMax.x &&
                               dstPixel.y >= letterboxMin.y && dstPixel.y < letterboxMax.y;

    if (insideContent) {
        const float2 uv = (dstPixel - letterboxMin + 0.5F) / scaledSize;
        const float4 sampled = inputTexture.SampleLevel(linearClampSampler, uv, 0.0F);
        const float3 rgbFromBgra = float3(sampled.b, sampled.g, sampled.r);
        const float3 rgb255 = saturate(rgbFromBgra) * 255.0F;
        rgb = rgb255 * inv255;
    }

    const uint linearIndex = dispatchThreadId.y * dstWidth + dispatchThreadId.x;
    const uint planeSize = dstWidth * dstHeight;

    outputTensor[linearIndex] = rgb.r;
    outputTensor[planeSize + linearIndex] = rgb.g;
    outputTensor[(2 * planeSize) + linearIndex] = rgb.b;
}
