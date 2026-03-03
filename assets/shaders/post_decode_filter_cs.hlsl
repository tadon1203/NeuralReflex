StructuredBuffer<float> rawOutput : register(t0);

RWStructuredBuffer<float4> candidateBox : register(u0);
RWStructuredBuffer<float2> candidateScoreClass : register(u1);
RWStructuredBuffer<uint> candidateCount : register(u2);

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

float readValue(uint anchorIndex, uint attributeIndex) {
    if (layoutFlag == 0) {
        // Attribute major: [attributes, anchors]
        return rawOutput[(attributeIndex * anchorCount) + anchorIndex];
    }
    // Anchor major: [anchors, attributes]
    return rawOutput[(anchorIndex * attributeCount) + attributeIndex];
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID) {
    const uint anchorIndex = dispatchThreadId.x;
    if (anchorIndex >= anchorCount) {
        return;
    }

    const uint availableClassCount =
        attributeCount > classStartIndex ? (attributeCount - classStartIndex) : 0;
    const uint activeClassCount = min(classCount, availableClassCount);
    if (activeClassCount == 0) {
        return;
    }

    float objectness = 1.0F;
    if (useObjectness != 0) {
        objectness = readValue(anchorIndex, 4);
    }

    float classScore = readValue(anchorIndex, classStartIndex);
    uint classId = 0;

    for (uint i = 1; i < activeClassCount; ++i) {
        const float candidateClassScore = readValue(anchorIndex, classStartIndex + i);
        if (candidateClassScore > classScore) {
            classScore = candidateClassScore;
            classId = i;
        }
    }

    const float finalScore = objectness * classScore;
    if (finalScore < scoreThreshold) {
        return;
    }

    uint outputIndex = 0;
    InterlockedAdd(candidateCount[0], 1, outputIndex);
    if (outputIndex >= anchorCount) {
        return;
    }

    candidateBox[outputIndex] = float4(readValue(anchorIndex, 0) * inputWidth,
                                       readValue(anchorIndex, 1) * inputHeight,
                                       readValue(anchorIndex, 2) * inputWidth,
                                       readValue(anchorIndex, 3) * inputHeight);
    candidateScoreClass[outputIndex] = float2(finalScore, asfloat(classId));
}
