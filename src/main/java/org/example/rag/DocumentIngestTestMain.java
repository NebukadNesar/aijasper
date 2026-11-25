package org.example.rag;

import org.example.embedding.MiniLmEmbeddingService;
import org.example.embedding.TokenizerService;

import java.nio.file.Path;
import java.util.List;

public class DocumentIngestTestMain {

     static void main(String[] args) throws Exception {
        Path vocabPath = Path.of("resources", "vocab.txt");
        Path modelPath = Path.of("resources", "onnx/model.onnx");
        Path pdfPath   = Path.of("resources", "deeplearningbook.pdf");

        PdfChunkLoader loader = new PdfChunkLoader();
        List<PdfChunkLoader.DocumentChunk> chunks = loader.loadAndChunk(pdfPath, 120, 25);
        System.out.println("Oluşan parça sayısı: " + chunks.size());

        TokenizerService tokenizer = new TokenizerService(vocabPath);
        try (MiniLmEmbeddingService embedder = new MiniLmEmbeddingService(modelPath)) {
            DocumentIngestionService ingestionService = new DocumentIngestionService(tokenizer, embedder);
            List<DocumentIngestionService.IndexedDocumentChunk> indexedChunks = ingestionService.ingest(chunks);

            System.out.println("İndekslenen parça sayısı: " + indexedChunks.size());
            if (!indexedChunks.isEmpty()) {
                DocumentIngestionService.IndexedDocumentChunk first = indexedChunks.get(0);
                System.out.println("İlk parça sayfa/index : " + first.pageNumber() + " / " + first.chunkIndex());
                System.out.println("Vektör boyutu         : " + first.embedding().length);
            }

            DocumentRetriever retriever = new DocumentRetriever(tokenizer, embedder);
            String question = "Logistic sigmoid";

            List<DocumentRetriever.RetrievedChunk> topMatches =  retriever.retrieveTopK(question, indexedChunks, 3);

            for (DocumentRetriever.RetrievedChunk match : topMatches) {
                System.out.printf("Skor: %.4f | Sayfa: %d | Chunk #%d%n",
                        match.score(), match.pageNumber(), match.chunkIndex());
                System.out.println(match.text().replaceAll("\\s+", " "));
                System.out.println("--------------------------------------------------");
            }
        }
    }
}