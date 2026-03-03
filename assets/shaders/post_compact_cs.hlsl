StructuredBuffer<float4> candidateBox : register(t0);
StructuredBuffer<float2> candidateScoreClass : register(t1);
StructuredBuffer<uint> candidateCount : register(t2);
StructuredBuffer<uint> suppressed : register(t3);

RWStructuredBuffer<float4> finalBox : register(u0);
RWStructuredBuffer<float2> finalScoreClass : register(u1);
RWStructuredBuffer<uint> finalCount : register(u2);

cbuffer PostConstants : register(b0) {
    uint anchorCount;
    uint attributeCount;
    uint layoutFlag;
    uint classCount;
    uint useObjectness;
    uint classStartIndex;
    uint maxDetections;
    uint reserved0;
    float inputWidth;
    float inputHeight;
    float scoreThreshold;
    float nmsIouThreshold;
};

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID) {
    const uint i = dispatchThreadId.x;
    if (i >= anchorCount) {
        return;
    }

    const uint count = candidateCount[0];
    if (i >= count || suppressed[i] != 0) {
        return;
    }

    uint outputIndex = 0;
    InterlockedAdd(finalCount[0], 1, outputIndex);
    if (outputIndex >= maxDetections) {
        return;
    }

    finalBox[outputIndex] = candidateBox[i];
    finalScoreClass[outputIndex] = candidateScoreClass[i];
}
