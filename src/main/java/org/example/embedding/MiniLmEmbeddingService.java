package org.example.embedding;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;

import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public class MiniLmEmbeddingService implements AutoCloseable {

    public static final int EMBEDDING_DIM = 384;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> inputNames;

    public MiniLmEmbeddingService(Path modelPath) throws OrtException {
        Objects.requireNonNull(modelPath, "modelPath cannot be null");

        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        // Optional CPU tuning:
        // options.setIntraOpNumThreads(4);

        this.session = env.createSession(modelPath.toString(), options);
        this.inputNames = session.getInputNames();
    }

    public float[] embed(EncodedInput encoded) throws OrtException {
        Objects.requireNonNull(encoded, "encoded cannot be null");
        return embed(encoded.inputIds(), encoded.attentionMask(), encoded.tokenTypeIds());
    }

    public float[] embed(long[] inputIds, long[] attentionMask, long[] tokenTypeIds) throws OrtException {
        Objects.requireNonNull(inputIds, "inputIds cannot be null");
        Objects.requireNonNull(attentionMask, "attentionMask cannot be null");

        int seqLen = inputIds.length;
        if (seqLen == 0) {
            throw new IllegalArgumentException("inputIds cannot be empty");
        }
        if (attentionMask.length != seqLen) {
            throw new IllegalArgumentException("attentionMask length must match inputIds length");
        }

        long[] shape = new long[]{1L, seqLen};
        Map<String, OnnxTensor> inputs = new HashMap<>();

        try {
            if (inputNames.contains("input_ids")) {
                inputs.put("input_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), shape));
            }
            if (inputNames.contains("attention_mask")) {
                inputs.put("attention_mask", OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), shape));
            }
            if (inputNames.contains("token_type_ids")) {
                long[] effectiveTokenTypeIds = (tokenTypeIds != null && tokenTypeIds.length == seqLen)
                        ? tokenTypeIds
                        : new long[seqLen];
                inputs.put("token_type_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(effectiveTokenTypeIds), shape));
            }

            try (Result result = session.run(inputs)) {
                // Expected output shape: [1, seqLen, 384]
                Object rawValue = result.get(0).getValue();
                float[][][] tokenEmbeddings = (float[][][]) rawValue;

                float[] pooled = meanPool(tokenEmbeddings[0], attentionMask);
                return l2Normalize(pooled);
            }
        } finally {
            for (OnnxTensor tensor : inputs.values()) {
                tensor.close();
            }
        }
    }

    private float[] meanPool(float[][] tokenMatrix, long[] attentionMask) {
        float[] sum = new float[EMBEDDING_DIM];
        float activeTokens = 0f;

        int tokensToPool = Math.min(tokenMatrix.length, attentionMask.length);
        for (int t = 0; t < tokensToPool; t++) {
            if (attentionMask[t] != 0L) {
                activeTokens += 1f;
                float[] tokenVec = tokenMatrix[t];
                for (int d = 0; d < EMBEDDING_DIM; d++) {
                    sum[d] += tokenVec[d];
                }
            }
        }

        float divisor = Math.max(activeTokens, 1e-9f);
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            sum[d] /= divisor;
        }
        return sum;
    }

    private float[] l2Normalize(float[] vector) {
        double sumSquares = 0.0;
        for (float v : vector) {
            sumSquares += (double) v * v;
        }

        double norm = Math.sqrt(sumSquares);
        if (norm < 1e-12) {
            return vector;
        }

        float[] normalized = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = (float) (vector[i] / norm);
        }
        return normalized;
    }

    @Override
    public void close() throws OrtException {
        session.close();
        env.close();
    }
}