package org.example.rag;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class PdfChunkLoader {

    public record DocumentChunk(
            int pageNumber,
            int chunkIndex,
            String text
    ) {
    }

    public List<DocumentChunk> loadAndChunk(Path pdfPath, int wordsPerChunk, int overlapWords) throws IOException {
        Objects.requireNonNull(pdfPath, "pdfPath cannot be null");
        if (wordsPerChunk <= 0) {
            throw new IllegalArgumentException("wordsPerChunk must be greater than 0");
        }
        if (overlapWords < 0 || overlapWords >= wordsPerChunk) {
            throw new IllegalArgumentException("overlapWords must be >= 0 and < wordsPerChunk");
        }

        List<DocumentChunk> chunks = new ArrayList<>();

        try (PDDocument document = Loader.loadPDF(pdfPath.toFile())) {
            PDFTextStripper stripper = new PDFTextStripper();
            int totalPages = document.getNumberOfPages();
            int globalIndex = 0;

            for (int page = 1; page <= totalPages; page++) {
                stripper.setStartPage(page);
                stripper.setEndPage(page);

                String pageText = stripper.getText(document).trim();
                if (pageText.isEmpty()) {
                    continue;
                }

                List<String> pieceList = splitByWords(pageText, wordsPerChunk, overlapWords);
                for (String piece : pieceList) {
                    chunks.add(new DocumentChunk(page, globalIndex++, piece));
                }
            }
        }

        return chunks;
    }

    private List<String> splitByWords(String text, int wordsPerChunk, int overlapWords) {
        String[] words = text.split("\\s+");
        List<String> pieces = new ArrayList<>();

        if (words.length <= wordsPerChunk) {
            pieces.add(String.join(" ", words));
            return pieces;
        }

        int step = wordsPerChunk - overlapWords;
        for (int start = 0; start < words.length; start += step) {
            int end = Math.min(words.length, start + wordsPerChunk);
            pieces.add(String.join(" ", Arrays.copyOfRange(words, start, end)));
            if (end == words.length) {
                break;
            }
        }

        return pieces;
    }
}