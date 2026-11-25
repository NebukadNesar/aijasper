package org.example.embedding;

public record EncodedInput(
        long[] inputIds,
        long[] attentionMask,
        long[] tokenTypeIds
) {
}