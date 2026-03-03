StructuredBuffer<float4> candidateBox : register(t0);
StructuredBuffer<float2> candidateScoreClass : register(t1);
StructuredBuffer<uint> candidateCount : register(t2);

RWStructuredBuffer<uint> suppressed : register(u0);

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

float computeIou(float4 a, float4 b) {
    const float ax1 = a.x - (a.z * 0.5F);
    const float ay1 = a.y - (a.w * 0.5F);
    const float ax2 = a.x + (a.z * 0.5F);
    const float ay2 = a.y + (a.w * 0.5F);

    const float bx1 = b.x - (b.z * 0.5F);
    const float by1 = b.y - (b.w * 0.5F);
    const float bx2 = b.x + (b.z * 0.5F);
    const float by2 = b.y + (b.w * 0.5F);

    const float interX1 = max(ax1, bx1);
    const float interY1 = max(ay1, by1);
    const float interX2 = min(ax2, bx2);
    const float interY2 = min(ay2, by2);

    const float interW = max(0.0F, interX2 - interX1);
    const float interH = max(0.0F, interY2 - interY1);
    const float interArea = interW * interH;

    const float areaA = max(0.0F, ax2 - ax1) * max(0.0F, ay2 - ay1);
    const float areaB = max(0.0F, bx2 - bx1) * max(0.0F, by2 - by1);
    const float unionArea = areaA + areaB - interArea;
    if (unionArea <= 0.0F) {
        return 0.0F;
    }

    return interArea / unionArea;
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID) {
    const uint i = dispatchThreadId.x;
    if (i >= anchorCount) {
        return;
    }

    const uint count = candidateCount[0];
    if (i >= count) {
        return;
    }

    suppressed[i] = 0;

    const float4 boxI = candidateBox[i];
    const float2 scoreClassI = candidateScoreClass[i];
    const float scoreI = scoreClassI.x;
    const uint classI = asuint(scoreClassI.y);

    for (uint j = 0; j < count; ++j) {
        if (j == i) {
            continue;
        }

        const float2 scoreClassJ = candidateScoreClass[j];
        const float scoreJ = scoreClassJ.x;
        const uint classJ = asuint(scoreClassJ.y);
        if (classJ != classI || scoreJ <= scoreI) {
            continue;
        }

        if (computeIou(boxI, candidateBox[j]) > nmsIouThreshold) {
            suppressed[i] = 1;
            return;
        }
    }
}
