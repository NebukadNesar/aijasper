package org.example.runner;

import org.example.asset.ModelAssetDownloader;
import org.example.embedding.MiniLmEmbeddingService;
import org.example.embedding.TokenizerService;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

public class MiniLLMService {
    static void download() {
        ModelAssetDownloader downloader = new ModelAssetDownloader(Path.of("resources"));
        try {
            downloader.downloadIfMissing();
            downloader.checkStatus();
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    static void main(String[] args) throws Exception {
        Path vocabPath = Path.of("resources", "vocab.txt");
        Path modelPath = Path.of("resources", "onnx", "model.onnx");

        TokenizerService tokenizer = new TokenizerService(vocabPath);

        try (MiniLmEmbeddingService embedder = new MiniLmEmbeddingService(modelPath)) {
            float[] emb1 = embedder.embed(tokenizer.encode("How do I run ONNX models in Java?"));
            float[] emb2 = embedder.embed(tokenizer.encode("Executing ONNX inference on a JVM backend."));
            float[] emb3 = embedder.embed(tokenizer.encode("Baking sourdough bread in a cast iron pot."));

            System.out.println("Embedding length : " + emb1.length);
            System.out.println("First 6 values   : " + Arrays.toString(Arrays.copyOf(emb1, 6)));
            System.out.printf("Cosine (1 vs 2)  : %.4f%n", cosineSimilarity(emb1, emb2));
            System.out.printf("Cosine (1 vs 3)  : %.4f%n", cosineSimilarity(emb1, emb3));
        }
    }

    private static float cosineSimilarity(float[] a, float[] b) {
        double dot = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
        }
        return (float) dot;
    }
}
