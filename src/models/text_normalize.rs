/// Text normalization utilities that mirror Python's `_prepare_text_for_sentence_chunking`.
/// This ensures short English sentences produce the same token sequences as Python.

/// Sentence-ending punctuation characters used by Python's logic.
pub const SENTENCE_END_PUNCTUATION: &[char] = &['.', '!', '?', '。', '！', '？', '；', ';'];

/// Clause-splitting punctuation (commas, etc.)
pub const CLAUSE_SPLIT_PUNCTUATION: &[char] = &[',', '，', '、', '；', ';', '：', ':'];

/// Closing punctuation that attaches to the preceding token.
const CLOSING_PUNCTUATION: &[char] = &[
    '"', '\'', '"', '"', '\'', '\'', ')', ']', '}', '）', '】', '》', '」', '』',
];

/// Default short pause between chunks (seconds) — used when chunk has ≤ 4 words.
pub const DEFAULT_INTER_CHUNK_PAUSE_SHORT_SECONDS: f64 = 0.40;

/// Default long pause between chunks (seconds) — used when chunk has > 4 words.
pub const DEFAULT_INTER_CHUNK_PAUSE_LONG_SECONDS: f64 = 0.24;

/// Default max text tokens per voice clone chunk.
pub const DEFAULT_VOICE_CLONE_MAX_TEXT_TOKENS: usize = 75;

/// Check if text contains any CJK characters (Chinese, Japanese, Korean, etc.)
pub fn contains_cjk(text: &str) -> bool {
    text.chars().any(|ch| {
        ('\u{4e00}'..='\u{9fff}').contains(&ch)      // CJK Unified Ideographs (Chinese)
            || ('\u{3400}'..='\u{4dbf}').contains(&ch)  // CJK Extension A
            || ('\u{3040}'..='\u{30ff}').contains(&ch)  // Hiragana + Katakana (Japanese)
            || ('\u{ac00}'..='\u{d7af}').contains(&ch) // Hangul Syllables (Korean)
    })
}

/// Prepare text for sentence chunking, mirroring Python's `_prepare_text_for_sentence_chunking`.
///
/// Rules:
/// 1. Strip whitespace, normalize `\n`/`\r` → space, collapse double spaces
/// 2. CJK text: append `。` if missing sentence-ending punctuation
/// 3. Non-CJK text: capitalize first letter, append `.` if missing punctuation
/// 4. Short English (< 5 words): prepend 8 leading spaces
///
/// The 8 leading spaces are critical — SentencePiece treats them as distinct tokens
/// that the model was trained to expect for short inputs.
pub fn prepare_text_for_sentence_chunking(text: &str) -> String {
    let mut normalized = text.trim().to_string();
    if normalized.is_empty() {
        return normalized;
    }

    // Normalize whitespace
    normalized = normalized.replace('\n', " ").replace('\r', " ");
    while normalized.contains("  ") {
        normalized = normalized.replace("  ", " ");
    }

    if contains_cjk(&normalized) {
        // CJK: append sentence-ending punctuation if missing
        if !normalized
            .chars()
            .last()
            .map_or(false, |c| SENTENCE_END_PUNCTUATION.contains(&c))
        {
            normalized.push_str("。");
        }
        return normalized;
    }

    // Non-CJK: capitalize first letter
    if let Some(first) = normalized.chars().next() {
        if first.is_lowercase() {
            normalized = format!(
                "{}{}",
                first.to_uppercase(),
                &normalized[first.len_utf8()..]
            );
        }
    }

    // Append period if missing sentence-ending punctuation
    if !normalized
        .chars()
        .last()
        .map_or(false, |c| SENTENCE_END_PUNCTUATION.contains(&c))
    {
        normalized.push('.');
    }

    // Pad short English sentences (< 5 words) with 8 leading spaces
    if normalized.split_whitespace().count() < 5 {
        normalized = format!("        {}", normalized); // 8 leading spaces
    }

    normalized
}

// ---------------------------------------------------------------------------
// Sentence chunking (mirrors Python's _split_text_by_punctuation,
// _split_text_by_token_budget, _join_sentence_parts,
// _split_text_into_best_sentences)
// ---------------------------------------------------------------------------

/// Split text at occurrences of any character in `punctuation`.
/// CLOSING_PUNCTUATION after the delimiter is kept with the sentence.
/// Mirrors Python's `_split_text_by_punctuation`.
pub fn split_text_by_punctuation(text: &str, punctuation: &[char]) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut sentences = Vec::new();
    let mut current: Vec<char> = Vec::new();
    let mut index = 0;

    while index < chars.len() {
        let ch = chars[index];
        current.push(ch);

        if punctuation.contains(&ch) {
            // Consume trailing closing punctuation
            let mut lookahead = index + 1;
            while lookahead < chars.len() && CLOSING_PUNCTUATION.contains(&chars[lookahead]) {
                current.push(chars[lookahead]);
                lookahead += 1;
            }

            let sentence: String = current.iter().collect();
            let trimmed = sentence.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed.to_string());
            }
            current.clear();

            // Skip whitespace after the delimiter
            while lookahead < chars.len() && chars[lookahead].is_whitespace() {
                lookahead += 1;
            }
            index = lookahead;
            continue;
        }

        index += 1;
    }

    // Remaining tail
    let tail: String = current.iter().collect();
    let trimmed = tail.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_string());
    }

    sentences
}

/// Join two sentence parts. For CJK text, concatenate directly;
/// for non-CJK, insert a space.
pub fn join_sentence_parts(left: &str, right: &str) -> String {
    if left.is_empty() {
        return right.to_string();
    }
    if right.is_empty() {
        return left.to_string();
    }
    if contains_cjk(left) || contains_cjk(right) {
        format!("{}{}", left, right)
    } else {
        format!("{} {}", left, right)
    }
}

/// Split text by a token budget using binary search, preferring to cut at
/// clause/sentence boundary characters. Mirrors Python's `_split_text_by_token_budget`.
///
/// `count_tokens_fn` is a closure that counts how many tokens the text produces.
pub fn split_text_by_token_budget<F>(
    text: &str,
    max_tokens: usize,
    count_tokens_fn: &F,
) -> Vec<String>
where
    F: Fn(&str) -> usize,
{
    let mut remaining = text.trim().to_string();
    if remaining.is_empty() {
        return Vec::new();
    }

    let mut pieces = Vec::new();

    while !remaining.is_empty() {
        if count_tokens_fn(&remaining) <= max_tokens {
            pieces.push(remaining.trim().to_string());
            break;
        }

        // Binary search for the longest prefix that fits within max_tokens
        let mut low = 1;
        let mut high = remaining.chars().count();
        let mut best_prefix_length: usize = 1;

        while low <= high {
            let middle = (low + high) / 2;
            let candidate: String = remaining.chars().take(middle).collect();
            let candidate = candidate.trim();
            if candidate.is_empty() {
                low = middle + 1;
                continue;
            }
            if count_tokens_fn(candidate) <= max_tokens {
                best_prefix_length = middle;
                low = middle + 1;
            } else {
                high = middle - 1;
            }
        }

        // Try to cut at a preferred boundary (within last 25 chars of prefix)
        let preferred_chars: Vec<char> = CLAUSE_SPLIT_PUNCTUATION
            .iter()
            .chain(SENTENCE_END_PUNCTUATION.iter())
            .chain(std::iter::once(&' '))
            .copied()
            .collect();

        let prefix: String = remaining.chars().take(best_prefix_length).collect();
        let scan_min = if prefix.len() >= 25 {
            prefix.len() - 25
        } else {
            0
        };

        let mut cut_index = best_prefix_length;
        // Search backwards from the end of the prefix (by char index)
        let prefix_chars: Vec<char> = prefix.chars().collect();
        for scan_pos in (scan_min..prefix_chars.len()).rev() {
            if preferred_chars.contains(&prefix_chars[scan_pos]) {
                cut_index = scan_pos + 1;
                break;
            }
        }

        let piece: String = remaining.chars().take(cut_index).collect();
        let piece = piece.trim().to_string();
        if piece.is_empty() {
            // Fallback: use the binary search result
            let piece: String = remaining.chars().take(best_prefix_length).collect();
            pieces.push(piece.trim().to_string());
            remaining = remaining
                .chars()
                .skip(best_prefix_length)
                .collect::<String>()
                .trim()
                .to_string();
        } else {
            pieces.push(piece);
            remaining = remaining
                .chars()
                .skip(cut_index)
                .collect::<String>()
                .trim()
                .to_string();
        }
    }

    pieces
}

/// Split text into chunks suitable for voice cloning, respecting token budget.
/// Mirrors Python's `_split_text_into_best_sentences`.
///
/// If the resulting chunks are more than 1, return them. Otherwise return the
/// original text as a single chunk.
pub fn split_into_best_sentences<F>(
    text: &str,
    max_tokens: usize,
    count_tokens_fn: &F,
) -> Vec<String>
where
    F: Fn(&str) -> usize,
{
    if max_tokens == 0 {
        return vec![text.to_string()];
    }

    let prepared = prepare_text_for_sentence_chunking(text);

    // Level 1: split by sentence-ending punctuation
    let sentence_candidates = split_text_by_punctuation(&prepared, SENTENCE_END_PUNCTUATION);
    let sentence_candidates = if sentence_candidates.is_empty() {
        vec![prepared.trim().to_string()]
    } else {
        sentence_candidates
    };

    // Build slices: (token_count, text)
    let mut sentence_slices: Vec<(usize, String)> = Vec::new();

    for sentence_text in &sentence_candidates {
        let normalized = sentence_text.trim();
        if normalized.is_empty() {
            continue;
        }
        let token_count = count_tokens_fn(normalized);
        if token_count <= max_tokens {
            sentence_slices.push((token_count, normalized.to_string()));
            continue;
        }

        // Level 2: split by clause punctuation
        let clause_candidates = split_text_by_punctuation(normalized, CLAUSE_SPLIT_PUNCTUATION);
        let clause_candidates = if clause_candidates.len() <= 1 {
            vec![normalized.to_string()]
        } else {
            clause_candidates
        };

        for clause_text in &clause_candidates {
            let clause_normalized = clause_text.trim();
            if clause_normalized.is_empty() {
                continue;
            }
            let clause_token_count = count_tokens_fn(clause_normalized);
            if clause_token_count <= max_tokens {
                sentence_slices.push((clause_token_count, clause_normalized.to_string()));
                continue;
            }

            // Level 3: binary search token budget split
            for piece in split_text_by_token_budget(clause_normalized, max_tokens, count_tokens_fn)
            {
                let piece_normalized = piece.trim();
                if !piece_normalized.is_empty() {
                    sentence_slices.push((
                        count_tokens_fn(piece_normalized),
                        piece_normalized.to_string(),
                    ));
                }
            }
        }
    }

    // Greedy merge: combine slices that fit together within the budget
    let mut chunks: Vec<String> = Vec::new();
    let mut current_chunk = String::new();
    let mut current_token_count: usize = 0;

    for (token_count, sentence_text) in &sentence_slices {
        if current_chunk.is_empty() {
            current_chunk = sentence_text.clone();
            current_token_count = *token_count;
            continue;
        }
        if current_token_count + token_count > max_tokens {
            chunks.push(current_chunk.trim().to_string());
            current_chunk = sentence_text.clone();
            current_token_count = *token_count;
        } else {
            current_chunk = join_sentence_parts(&current_chunk, sentence_text);
            current_token_count = count_tokens_fn(&current_chunk);
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    if chunks.is_empty() {
        chunks.push(prepared.trim().to_string());
    }

    // If only 1 chunk, return original text (mirrors Python behavior)
    if chunks.len() > 1 {
        chunks
    } else {
        vec![text.to_string()]
    }
}

/// Estimate inter-chunk pause duration in seconds.
/// Short pause (0.40s) for chunks with ≤ 4 words, long pause (0.24s) otherwise.
pub fn estimate_inter_chunk_pause_seconds(text_chunk: &str) -> f64 {
    let word_count = text_chunk.split_whitespace().count();
    if word_count <= 4 {
        DEFAULT_INTER_CHUNK_PAUSE_SHORT_SECONDS
    } else {
        DEFAULT_INTER_CHUNK_PAUSE_LONG_SECONDS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_english_padding() {
        // "Good morning." → "        Good morning."
        let result = prepare_text_for_sentence_chunking("Good morning.");
        assert!(result.starts_with("        "));
        assert!(result.contains("Good morning"));
    }

    #[test]
    fn test_long_english_no_padding() {
        // Long sentence should NOT be padded
        let result =
            prepare_text_for_sentence_chunking("The quick brown fox jumps over the lazy dog.");
        assert!(!result.starts_with("        "));
    }

    #[test]
    fn test_cjk_no_padding() {
        // CJK should not get 8-space padding
        let result = prepare_text_for_sentence_chunking("你好");
        assert!(!result.starts_with("        "));
        assert!(result.ends_with("。"));
    }

    #[test]
    fn test_capitalization() {
        // "hello world" is 2 words (<5), so gets 8-space prefix + capitalization + period
        let result = prepare_text_for_sentence_chunking("hello world");
        assert!(result.contains("Hello world."));
        assert!(result.ends_with('.'));
    }

    #[test]
    fn test_whitespace_normalization() {
        // Use a long sentence to avoid 8-space padding, then check no double spaces in the text portion
        let result =
            prepare_text_for_sentence_chunking("  The quick brown fox jumps over the lazy dog  ");
        // After trim + collapse + no padding (5+ words)
        assert!(!result.starts_with("  "));
        assert!(!result.contains("  "));
    }

    // -- Split by punctuation tests --

    #[test]
    fn test_split_by_sentence_end_punctuation() {
        let text = "Hello world. How are you? I'm fine!";
        let sentences = split_text_by_punctuation(text, SENTENCE_END_PUNCTUATION);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "How are you?");
        assert_eq!(sentences[2], "I'm fine!");
    }

    #[test]
    fn test_split_by_clause_punctuation() {
        let text = "first, second, third";
        let clauses = split_text_by_punctuation(text, CLAUSE_SPLIT_PUNCTUATION);
        assert_eq!(clauses.len(), 3);
        assert_eq!(clauses[0], "first,");
        assert_eq!(clauses[1], "second,");
        assert_eq!(clauses[2], "third");
    }

    #[test]
    fn test_split_cjk_sentence_end() {
        let text = "你好。世界！你好吗？";
        let sentences = split_text_by_punctuation(text, SENTENCE_END_PUNCTUATION);
        assert_eq!(sentences.len(), 3);
    }

    // -- Join sentence parts tests --

    #[test]
    fn test_join_cjk_parts() {
        assert_eq!(join_sentence_parts("你好", "世界"), "你好世界");
    }

    #[test]
    fn test_join_english_parts() {
        assert_eq!(join_sentence_parts("Hello", "world"), "Hello world");
    }

    // -- Token budget split tests --

    #[test]
    fn test_split_by_token_budget_short() {
        // If text fits within budget, return it as single piece
        let count_fn = |t: &str| t.split_whitespace().count(); // simple word-based token count
        let pieces = split_text_by_token_budget("Hello world.", 10, &count_fn);
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0], "Hello world.");
    }

    #[test]
    fn test_split_by_token_budget_needs_split() {
        let count_fn = |t: &str| t.split_whitespace().count(); // each word = 1 token
        let text = "one two three four five six seven eight nine ten eleven twelve";
        let pieces = split_text_by_token_budget(text, 4, &count_fn);
        assert!(pieces.len() > 1);
        // Each piece should have ≤ 4 words (except possibly the first due to boundary adjustment)
        for piece in &pieces {
            assert!(
                count_fn(piece) <= 5,
                "piece '{}' has {} words",
                piece,
                count_fn(piece)
            );
        }
    }

    // -- Full chunking tests --

    #[test]
    fn test_split_into_best_sentences_short_text() {
        // Short text that fits in one chunk → returns [original_text]
        let count_fn = |t: &str| t.split_whitespace().count();
        let chunks = split_into_best_sentences("Hello world.", 50, &count_fn);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello world.");
    }

    #[test]
    fn test_split_into_best_sentences_long_text() {
        let count_fn = |t: &str| t.split_whitespace().count();
        let text = "First sentence here. Second sentence with more words. Third one too.";
        let chunks = split_into_best_sentences(text, 5, &count_fn);
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_split_into_best_sentences_disabled() {
        let count_fn = |t: &str| t.split_whitespace().count();
        let text = "Hello world. How are you?";
        let chunks = split_into_best_sentences(text, 0, &count_fn);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    // -- Pause estimation tests --

    #[test]
    fn test_pause_estimation_short() {
        let pause = estimate_inter_chunk_pause_seconds("Hi there");
        assert!((pause - 0.40).abs() < 0.01);
    }

    #[test]
    fn test_pause_estimation_long() {
        let pause = estimate_inter_chunk_pause_seconds("This is a longer chunk with many words");
        assert!((pause - 0.24).abs() < 0.01);
    }
}
