package org.example.rag;

import ai.onnxruntime.OrtException;

import org.example.embedding.EncodedInput;
import org.example.embedding.MiniLmEmbeddingService;
import org.example.embedding.TokenizerService;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

public class DocumentRetriever {

    public record RetrievedChunk(
            int pageNumber,
            int chunkIndex,
            String text,
            float score
    ) {}

    private final TokenizerService tokenizer;
    private final MiniLmEmbeddingService embeddingService;

    public DocumentRetriever(TokenizerService tokenizer, MiniLmEmbeddingService embeddingService) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer cannot be null");
        this.embeddingService = Objects.requireNonNull(embeddingService, "embeddingService cannot be null");
    }

    public List<RetrievedChunk> retrieveTopK(
            String question,
            List<DocumentIngestionService.IndexedDocumentChunk> indexedChunks,
            int topK
    ) throws OrtException {
        if (topK <= 0 || indexedChunks == null || indexedChunks.isEmpty()) {
            return List.of();
        }

        EncodedInput encodedQuestion = tokenizer.encode(question);
        float[] questionVector = embeddingService.embed(encodedQuestion);

        List<RetrievedChunk> scored = new ArrayList<>(indexedChunks.size());
        for (DocumentIngestionService.IndexedDocumentChunk chunk : indexedChunks) {
            float score = dotProduct(questionVector, chunk.embedding());
            scored.add(new RetrievedChunk(
                    chunk.pageNumber(),
                    chunk.chunkIndex(),
                    chunk.text(),
                    score
            ));
        }

        scored.sort(Comparator.comparingDouble(RetrievedChunk::score).reversed());

        int limit = Math.min(topK, scored.size());
        return List.copyOf(scored.subList(0, limit));
    }

    public static float dotProduct(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "Vector dimensions must match: " + a.length + " vs " + b.length
            );
        }
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += (double) a[i] * b[i];
        }
        return (float) sum;
    }
}