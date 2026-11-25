package org.example.asset;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.Objects;

public class ModelAssetDownloader {

    private static final String BASE_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main";

    private static final String ONNX_REMOTE_PATH = "onnx/model.onnx";
    private static final String VOCAB_REMOTE_PATH = "vocab.txt";

    private final Path targetDir;
    private final HttpClient httpClient;

    public ModelAssetDownloader(Path targetDir) {
        this.targetDir = Objects.requireNonNull(targetDir, "targetDir cannot be null");
        this.httpClient = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.ALWAYS)
                .connectTimeout(Duration.ofSeconds(15))
                .build();
    }

    public Path getModelPath() {
        return targetDir.resolve("onnx").resolve("model.onnx");
    }

    public Path getVocabPath() {
        return targetDir.resolve("vocab.txt");
    }

    public boolean isModelReady() {
        return isValidFile(getModelPath()) && isValidFile(getVocabPath());
    }

    public void checkStatus() {
        Path modelPath = getModelPath();
        Path vocabPath = getVocabPath();

        System.out.println("Target directory : " + targetDir.toAbsolutePath());
        System.out.println("model.onnx exists: " + Files.exists(modelPath)
                + (Files.exists(modelPath) ? " (" + formatSize(modelPath) + ")" : ""));
        System.out.println("vocab.txt exists : " + Files.exists(vocabPath)
                + (Files.exists(vocabPath) ? " (" + formatSize(vocabPath) + ")" : ""));
        System.out.println("Ready for inference: " + isModelReady());
    }

    public void downloadIfMissing() throws IOException, InterruptedException {
        Files.createDirectories(targetDir);

        downloadFileIfMissing(ONNX_REMOTE_PATH, getModelPath());
        downloadFileIfMissing(VOCAB_REMOTE_PATH, getVocabPath());
    }

    private void downloadFileIfMissing(String remotePath, Path localPath) throws IOException, InterruptedException {
        if (isValidFile(localPath)) {
            System.out.println("Already present, skipping: " + localPath.toAbsolutePath());
            return;
        }

        String url = BASE_URL + "/" + remotePath;
        System.out.println("Downloading from: " + url);
        System.out.println("Saving to       : " + localPath.toAbsolutePath());

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofMinutes(5))
                .GET()
                .build();

        HttpResponse<InputStream> response = httpClient.send(
                request, HttpResponse.BodyHandlers.ofInputStream()
        );

        if (response.statusCode() != 200) {
            throw new IOException("Failed to download " + url + " -> HTTP " + response.statusCode());
        }

        Path tempFile = Files.createTempFile(targetDir, ".download-", ".part");
        try (InputStream in = response.body()) {
            Files.copy(in, tempFile, StandardCopyOption.REPLACE_EXISTING);

            Path parentDir = localPath.getParent();
            if (parentDir != null) {
                Files.createDirectories(parentDir);
            }

            Files.move(tempFile, localPath, StandardCopyOption.REPLACE_EXISTING);
        } catch (Exception e) {
            Files.deleteIfExists(tempFile);
            throw e;
        }

        System.out.println("Saved: " + localPath.toAbsolutePath() + " (" + formatSize(localPath) + ")");
    }


    private boolean isValidFile(Path path) {
        try {
            return Files.isRegularFile(path) && Files.size(path) > 0L;
        } catch (IOException e) {
            return false;
        }
    }

    private String formatSize(Path path) {
        try {
            long bytes = Files.size(path);
            double mb = bytes / (1024.0 * 1024.0);
            return String.format("%.2f MB", mb);
        } catch (IOException e) {
            return "unknown size";
        }
    }
}