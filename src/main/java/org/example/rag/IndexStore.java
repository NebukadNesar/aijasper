package org.example.rag;

import ai.onnxruntime.OrtException;
import org.example.rag.DocumentIngestionService;
import org.example.rag.DocumentIngestionService.IndexedDocumentChunk;
import org.example.rag.PdfChunkLoader;
import org.example.rag.PdfChunkLoader.DocumentChunk;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class IndexStore {

    private static final int MAGIC = 0x52414749; // "RAGI"
    private static final int VERSION = 1;
    private static final Path DEFAULT_RESOURCES_DIR =
            Path.of("..", "demo", "AIJaper", "resources");

    private final Path resourcesDir;

    public IndexStore() {
        this(DEFAULT_RESOURCES_DIR);
    }

    public IndexStore(Path resourcesDir) {
        this.resourcesDir = Objects.requireNonNull(resourcesDir, "resourcesDir cannot be null");
    }

    public Path getResourcesDir() {
        return resourcesDir;
    }

    public Path resolveIndexPath(String fileName) {
        Objects.requireNonNull(fileName, "fileName cannot be null");
        return resourcesDir.resolve(fileName).normalize();
    }

    public List<IndexedDocumentChunk> loadOrBuildFromResources(
            String indexFileName,
            Path pdfPath,
            int wordsPerChunk,
            int overlap,
            PdfChunkLoader chunkLoader,
            DocumentIngestionService ingestionService
    ) throws IOException, OrtException {
        Objects.requireNonNull(indexFileName, "indexFileName cannot be null");
        Objects.requireNonNull(pdfPath, "pdfPath cannot be null");
        Objects.requireNonNull(chunkLoader, "chunkLoader cannot be null");
        Objects.requireNonNull(ingestionService, "ingestionService cannot be null");

        Path targetPath = resolveIndexPath(indexFileName);
        if (exists(targetPath)) {
            return load(targetPath);
        }

        List<DocumentChunk> rawChunks = chunkLoader.loadAndChunk(pdfPath, wordsPerChunk, overlap);
        List<IndexedDocumentChunk> indexedChunks = ingestionService.ingest(rawChunks);
        save(targetPath, indexedChunks);
        return indexedChunks;
    }

    public void saveToResources(String fileName, List<IndexedDocumentChunk> chunks) throws IOException {
        save(resolveIndexPath(fileName), chunks);
    }

    public List<IndexedDocumentChunk> loadFromResources(String fileName) throws IOException {
        return load(resolveIndexPath(fileName));
    }

    public boolean existsInResources(String fileName) {
        return exists(resolveIndexPath(fileName));
    }

    public void save(Path indexPath, List<IndexedDocumentChunk> chunks) throws IOException {
        Objects.requireNonNull(indexPath, "indexPath cannot be null");
        Objects.requireNonNull(chunks, "chunks cannot be null");

        Path normalizedPath = indexPath.normalize();
        Path parent = normalizedPath.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(normalizedPath)))) {
            out.writeInt(MAGIC);
            out.writeInt(VERSION);
            out.writeInt(chunks.size());

            for (IndexedDocumentChunk chunk : chunks) {
                out.writeInt(chunk.pageNumber());
                out.writeInt(chunk.chunkIndex());

                byte[] textBytes = chunk.text().getBytes(StandardCharsets.UTF_8);
                out.writeInt(textBytes.length);
                out.write(textBytes);

                float[] embedding = chunk.embedding();
                Objects.requireNonNull(embedding, "chunk embedding cannot be null");
                out.writeInt(embedding.length);
                for (float value : embedding) {
                    out.writeFloat(value);
                }
            }
            out.flush();
        }
    }

    public List<IndexedDocumentChunk> load(Path indexPath) throws IOException {
        Objects.requireNonNull(indexPath, "indexPath cannot be null");
        Path normalizedPath = indexPath.normalize();
        if (!Files.isRegularFile(normalizedPath)) {
            throw new IOException("Index file not found: " + normalizedPath);
        }

        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(normalizedPath)))) {
            int magic = in.readInt();
            if (magic != MAGIC) {
                throw new IOException("Invalid index file magic header in: " + normalizedPath);
            }

            int version = in.readInt();
            if (version != VERSION) {
                throw new IOException("Unsupported index version: " + version);
            }

            int chunkCount = in.readInt();
            if (chunkCount < 0) {
                throw new IOException("Corrupt chunk count in index: " + chunkCount);
            }

            List<IndexedDocumentChunk> chunks = new ArrayList<>(chunkCount);
            for (int i = 0; i < chunkCount; i++) {
                int pageNumber = in.readInt();
                int chunkIndex = in.readInt();

                int textLength = in.readInt();
                if (textLength < 0) {
                    throw new IOException("Corrupt text length at chunk #" + i);
                }
                byte[] textBytes = readExactBytes(in, textLength, i);
                String text = new String(textBytes, StandardCharsets.UTF_8);

                int embeddingLength = in.readInt();
                if (embeddingLength < 0) {
                    throw new IOException("Corrupt embedding length at chunk #" + i);
                }
                float[] embedding = new float[embeddingLength];
                for (int j = 0; j < embeddingLength; j++) {
                    embedding[j] = in.readFloat();
                }

                chunks.add(new IndexedDocumentChunk(pageNumber, chunkIndex, text, embedding));
            }

            return List.copyOf(chunks);
        }
    }

    public boolean exists(Path indexPath) {
        return indexPath != null && Files.isRegularFile(indexPath.normalize());
    }

    private static byte[] readExactBytes(DataInputStream in, int length, int chunkIndex) throws IOException {
        byte[] buffer = in.readNBytes(length);
        if (buffer.length != length) {
            throw new EOFException("Unexpected end of stream while reading chunk #" + chunkIndex);
        }
        return buffer;
    }
}