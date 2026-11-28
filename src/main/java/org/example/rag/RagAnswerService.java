package org.example.rag;

import org.example.rag.DocumentRetriever.RetrievedChunk;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Objects;

public class RagAnswerService {

    private final HttpClient httpClient;
    private final URI chatEndpoint;
    private final String model;

    public RagAnswerService(String ollamaBaseUrl, String model) {
        Objects.requireNonNull(ollamaBaseUrl, "ollamaBaseUrl cannot be null");
        this.model = Objects.requireNonNull(model, "model cannot be null");
        String normalizedBase = ollamaBaseUrl.endsWith("/")
                ? ollamaBaseUrl.substring(0, ollamaBaseUrl.length() - 1)
                : ollamaBaseUrl;
        this.chatEndpoint = URI.create(normalizedBase + "/api/chat");
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    public String answer(String question, List<RetrievedChunk> chunks)
            throws IOException, InterruptedException {
        if (chunks == null || chunks.isEmpty()) {
            return "Verilen doküman parçaları içinde bu soruyu yanıtlamak için yeterli bağlam bulunamadı.";
        }

        String context = buildContext(chunks);
        String systemPrompt = """
                You are a technical assistant answering questions strictly from the provided PDF context.
                Rules:
                1. Rely only on facts present in the CONTEXT blocks below.
                2. Do not guess or invent facts outside the context. If the answer is not in the context, state that clearly.
                3. Mention relevant page numbers when referencing information.
                4. Write a clear, concise, well-structured answer in the same language as the user's question.
                """;

        String userPrompt = """
                QUESTION:
                %s
                
                CONTEXT:
                %s
                """.formatted(question, context);
        String requestBody = """
                {
                  "model": "%s",
                  "stream": false,
                  "think": false,
                  "messages": [
                    {
                      "role": "system",
                      "content": "%s"
                    },
                    {
                      "role": "user",
                      "content": "%s"
                    }
                  ],
                  "options": {
                    "temperature": 0.2,
                    "num_ctx": 4096,
                    "num_predict": 400
                  }
                }
                """.formatted(
                escapeJson(model),
                escapeJson(systemPrompt),
                escapeJson(userPrompt)
        );


        HttpRequest request = HttpRequest.newBuilder(chatEndpoint)
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(330))
                .POST(HttpRequest.BodyPublishers.ofString(requestBody, StandardCharsets.UTF_8))
                .build();

        HttpResponse<String> response = httpClient.send(
                request,
                HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8)
        );

        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw new IOException("Ollama isteği başarısız oldu. HTTP " + response.statusCode()
                    + ": " + response.body());
        }

        return extractAssistantContent(response.body());
    }

    private String buildContext(List<RetrievedChunk> chunks) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < chunks.size(); i++) {
            RetrievedChunk chunk = chunks.get(i);
            String cleanedText = chunk.text().replaceAll("\\s+", " ").trim();
            sb.append("[Parça ").append(i + 1)
                    .append(" | Sayfa ").append(chunk.pageNumber())
                    .append(" | Skor ").append(String.format(java.util.Locale.US, "%.4f", chunk.score()))
                    .append("]\n")
                    .append(cleanedText)
                    .append("\n\n");
        }
        return sb.toString().trim();
    }

    private static String escapeJson(String value) {
        StringBuilder out = new StringBuilder(value.length());
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default -> {
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                }
            }
        }
        return out.toString();
    }

    // Ollama /api/chat yanıtından "message": { ..., "content": "..." } alanını ayıklar.
    // Projende Jackson/Gson varsa doğrudan JSON parser kullanman daha sağlam olur.
    private static String extractAssistantContent(String json) {
        int messageIdx = json.indexOf("\"message\"");
        if (messageIdx < 0) {
            throw new IllegalStateException("Beklenmeyen Ollama yanıtı: " + json);
        }
        int contentKeyIdx = json.indexOf("\"content\"", messageIdx);
        if (contentKeyIdx < 0) {
            throw new IllegalStateException("Yanıtta 'content' alanı bulunamadı: " + json);
        }
        int colonIdx = json.indexOf(':', contentKeyIdx + "\"content\"".length());
        int firstQuoteIdx = json.indexOf('"', colonIdx + 1);
        if (firstQuoteIdx < 0) {
            throw new IllegalStateException("Yanıttaki 'content' değeri okunamadı: " + json);
        }

        StringBuilder sb = new StringBuilder();
        boolean escaped = false;
        for (int i = firstQuoteIdx + 1; i < json.length(); i++) {
            char c = json.charAt(i);
            if (escaped) {
                switch (c) {
                    case '"' -> sb.append('"');
                    case '\\' -> sb.append('\\');
                    case '/' -> sb.append('/');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'u' -> {
                        if (i + 4 < json.length()) {
                            String hex = json.substring(i + 1, i + 5);
                            sb.append((char) Integer.parseInt(hex, 16));
                            i += 4;
                        }
                    }
                    default -> sb.append(c);
                }
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                return sb.toString();
            } else {
                sb.append(c);
            }
        }
        throw new IllegalStateException("JSON string sonu bulunamadı: " + json);
    }
}