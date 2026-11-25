package org.example.embedding;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

public class TokenizerService {

    private static final String PAD_TOKEN = "[PAD]";
    private static final String UNK_TOKEN = "[UNK]";
    private static final String CLS_TOKEN = "[CLS]";
    private static final String SEP_TOKEN = "[SEP]";
    private static final String SUBWORD_PREFIX = "##";

    private static final int DEFAULT_MAX_LENGTH = 128;
    private static final int MAX_WORD_CHARS = 100;

    private final Map<String, Integer> vocab;
    private final int maxLength;
    private final int padTokenId;
    private final int unkTokenId;
    private final int clsTokenId;
    private final int sepTokenId;

    public TokenizerService(Path vocabPath) throws IOException {
        this(vocabPath, DEFAULT_MAX_LENGTH);
    }

    public TokenizerService(Path vocabPath, int maxLength) throws IOException {
        Objects.requireNonNull(vocabPath, "vocabPath cannot be null");
        if (maxLength < 2) {
            throw new IllegalArgumentException("maxLength must be at least 2");
        }
        this.maxLength = maxLength;
        this.vocab = loadVocab(vocabPath);

        this.padTokenId = requireTokenId(PAD_TOKEN);
        this.unkTokenId = requireTokenId(UNK_TOKEN);
        this.clsTokenId = requireTokenId(CLS_TOKEN);
        this.sepTokenId = requireTokenId(SEP_TOKEN);
    }

    private Map<String, Integer> loadVocab(Path vocabPath) throws IOException {
        Map<String, Integer> map = new HashMap<>();
        try (BufferedReader reader = Files.newBufferedReader(vocabPath, StandardCharsets.UTF_8)) {
            String line;
            int index = 0;
            while ((line = reader.readLine()) != null) {
                if (!line.isEmpty()) {
                    map.putIfAbsent(line, index);
                }
                index++;
            }
        }
        if (map.isEmpty()) {
            throw new IOException("Vocabulary file is empty: " + vocabPath);
        }
        return map;
    }

    private int requireTokenId(String token) {
        Integer id = vocab.get(token);
        if (id == null) {
            throw new IllegalStateException("Missing required token in vocab: " + token);
        }
        return id;
    }

    public EncodedInput encode(String text) {
        String safeText = text == null ? "" : text;
        List<String> basicTokens = basicTokenize(safeText);

        int maxContentTokens = maxLength - 2; // [CLS] ve [SEP] hariç
        List<Integer> tokenIds = new ArrayList<>(maxLength);
        tokenIds.add(clsTokenId);

        int contentCount = 0;
        for (String word : basicTokens) {
            for (int subwordId : tokenizeWordPiece(word)) {
                if (contentCount >= maxContentTokens) {
                    break;
                }
                tokenIds.add(subwordId);
                contentCount++;
            }
            if (contentCount >= maxContentTokens) {
                break;
            }
        }

        tokenIds.add(sepTokenId);

        int size = tokenIds.size();
        long[] inputIds = new long[size];
        long[] attentionMask = new long[size];
        long[] tokenTypeIds = new long[size];

        for (int i = 0; i < size; i++) {
            inputIds[i] = tokenIds.get(i);
            attentionMask[i] = 1L;
            tokenTypeIds[i] = 0L;
        }

        return new EncodedInput(inputIds, attentionMask, tokenTypeIds);
    }

    private List<String> basicTokenize(String text) {
        String normalized = Normalizer.normalize(text, Normalizer.Form.NFKC)
                .toLowerCase(Locale.ROOT);

        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();

        for (int i = 0; i < normalized.length(); i++) {
            char ch = normalized.charAt(i);

            if (Character.isWhitespace(ch) || isControlChar(ch)) {
                flush(current, tokens);
            } else if (isPunctuation(ch)) {
                flush(current, tokens);
                tokens.add(String.valueOf(ch));
            } else {
                current.append(ch);
            }
        }
        flush(current, tokens);
        return tokens;
    }

    private void flush(StringBuilder current, List<String> tokens) {
        if (!current.isEmpty()) {
            tokens.add(current.toString());
            current.setLength(0);
        }
    }

    private List<Integer> tokenizeWordPiece(String token) {
        List<Integer> result = new ArrayList<>();
        if (token.isEmpty()) {
            return result;
        }
        if (token.length() > MAX_WORD_CHARS) {
            result.add(unkTokenId);
            return result;
        }

        int start = 0;
        boolean isFirst = true;

        while (start < token.length()) {
            int end = token.length();
            Integer matchedId = null;

            while (start < end) {
                String slice = token.substring(start, end);
                String candidate = isFirst ? slice : SUBWORD_PREFIX + slice;
                Integer id = vocab.get(candidate);
                if (id != null) {
                    matchedId = id;
                    break;
                }
                end--;
            }

            if (matchedId == null) {
                return List.of(unkTokenId);
            }

            result.add(matchedId);
            start = end;
            isFirst = false;
        }

        return result;
    }

    private boolean isControlChar(char ch) {
        if (ch == '\t' || ch == '\n' || ch == '\r') {
            return false;
        }
        int type = Character.getType(ch);
        return type == Character.CONTROL
                || type == Character.FORMAT
                || type == Character.SURROGATE
                || type == Character.PRIVATE_USE
                || type == Character.UNASSIGNED;
    }

    private boolean isPunctuation(char ch) {
        if ((ch >= 33 && ch <= 47)
                || (ch >= 58 && ch <= 64)
                || (ch >= 91 && ch <= 96)
                || (ch >= 123 && ch <= 126)) {
            return true;
        }
        int type = Character.getType(ch);
        return type == Character.DASH_PUNCTUATION
                || type == Character.START_PUNCTUATION
                || type == Character.END_PUNCTUATION
                || type == Character.CONNECTOR_PUNCTUATION
                || type == Character.OTHER_PUNCTUATION
                || type == Character.INITIAL_QUOTE_PUNCTUATION
                || type == Character.FINAL_QUOTE_PUNCTUATION;
    }

    public int getPadTokenId() {
        return padTokenId;
    }

    public int getMaxLength() {
        return maxLength;
    }
}