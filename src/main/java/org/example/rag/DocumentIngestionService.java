package org.example.rag;

import ai.onnxruntime.OrtException;
import org.example.embedding.EncodedInput;
import org.example.embedding.MiniLmEmbeddingService;
import org.example.embedding.TokenizerService;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class DocumentIngestionService {

    public record IndexedDocumentChunk(
            int pageNumber,
            int chunkIndex,
            String text,
            float[] embedding
    ) {
    }

    private final TokenizerService tokenizer;
    private final MiniLmEmbeddingService embeddingService;

    public DocumentIngestionService(TokenizerService tokenizer, MiniLmEmbeddingService embeddingService) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer cannot be null");
        this.embeddingService = Objects.requireNonNull(embeddingService, "embeddingService cannot be null");
    }

    public List<IndexedDocumentChunk> ingest(List<PdfChunkLoader.DocumentChunk> chunks) throws OrtException {
        Objects.requireNonNull(chunks, "chunks cannot be null");
        List<IndexedDocumentChunk> indexed = new ArrayList<>(chunks.size());

        for (PdfChunkLoader.DocumentChunk chunk : chunks) {
            EncodedInput encoded = tokenizer.encode(chunk.text());
            float[] vector = embeddingService.embed(encoded);
            indexed.add(new IndexedDocumentChunk(
                    chunk.pageNumber(),
                    chunk.chunkIndex(),
                    chunk.text(),
                    vector
            ));
        }

        return indexed;
    }
}