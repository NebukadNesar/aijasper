package org.example.rag;


import org.example.embedding.MiniLmEmbeddingService;
import org.example.embedding.TokenizerService;
import org.example.rag.DocumentRetriever.RetrievedChunk;

import java.nio.file.Path;
import java.util.List;

public class RagPipelineRunner {

    static void main(String[] args) throws Exception {
        Path vocabPath = Path.of("resources", "vocab.txt");
        Path onnxModelPath = Path.of("resources", "onnx/model.onnx");
        Path pdfPath   = Path.of("resources", "deeplearningbook.pdf");

        String ollamaUrl = "http://localhost:11434";
        String llmModel = "qwen3:8b";
        String question = "What is an embedding and how is it used in neural networks?";

        // 1. PDF'i oku ve parçalara ayır
        PdfChunkLoader chunkLoader = new PdfChunkLoader();
        TokenizerService tokenizer = new TokenizerService(vocabPath, 256);
        MiniLmEmbeddingService embeddingService = new MiniLmEmbeddingService(onnxModelPath);
        DocumentIngestionService ingestionService = new DocumentIngestionService(tokenizer, embeddingService);
        IndexStore indexStore = new IndexStore();
        List<DocumentIngestionService.IndexedDocumentChunk> indexedChunks = indexStore.loadOrBuildFromResources(
                "deeplearningbook-minilm.ragidx",
                pdfPath,
                180,
                30,
                chunkLoader,
                ingestionService
        );

        // 3. Soruya en yakın top-k parçaları getir
        DocumentRetriever retriever = new DocumentRetriever(tokenizer, embeddingService);
        List<RetrievedChunk> topChunks = retriever.retrieveTopK(question, indexedChunks, 5);


        System.out.println("=== BULUNAN KAYNAK PARÇALAR ===");
        for (RetrievedChunk chunk : topChunks) {
            System.out.printf("Sayfa %d | Skor %.4f%n | Text %s", chunk.pageNumber(), chunk.score(), chunk.text());
        }

        System.out.println(" --- Answering --- ");
        // 4. Seçilen parçaları LLM'e verip bağlama dayalı cevap üret
        RagAnswerService ragAnswerService = new RagAnswerService(ollamaUrl, llmModel);
        String finalAnswer = ragAnswerService.answer(question, topChunks);

        System.out.println("\n=== LLM CEVABI ===");
        System.out.println(finalAnswer);
    }
}