# Unified single-inference extraction mapped directly from Untitled (1) (1).ipynb

import numpy as np
import re
from collections import Counter
from scipy.stats import entropy as scipy_entropy, spearmanr
import spacy
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker

# ==========================================
# Helpers mapped natively
# ==========================================

word_pattern = re.compile(r'\b[a-zA-Z]+\b')
sentence_splitter = re.compile(r'[.!?]+')
header_pattern = re.compile(r'^(section|chapter|part)\s+\d+', re.IGNORECASE)

DISCOURSE_MARKERS = {'therefore', 'thus', 'hence', 'consequently', 'accordingly', 'as a result', 'for this reason', 'however', 'nevertheless', 'nonetheless', 'on the other hand', 'in contrast', 'alternatively', 'yet', 'still', 'moreover', 'furthermore', 'in addition', 'additionally', 'also', 'besides', 'similarly', 'likewise', 'notably', 'significantly', 'particularly', 'especially', 'importantly', 'indeed', 'clearly', 'for example', 'for instance', 'such as', 'including', 'firstly', 'secondly', 'finally', 'subsequently', 'then', 'next', 'in conclusion', 'to conclude', 'in summary', 'to summarize', 'overall', 'ultimately', 'in closing'}
SAFE_VOCAB = {'important', 'significant', 'essential', 'crucial', 'key', 'critical', 'fundamental', 'vital', 'necessary', 'valuable', 'useful', 'effective', 'efficient', 'beneficial', 'relevant', 'meaningful', 'appropriate', 'notable', 'various', 'numerous', 'several', 'different', 'many', 'some', 'certain', 'specific', 'general'}
IMPORTANCE_WORDS = {'important', 'crucial', 'significant', 'essential', 'key', 'critical', 'vital', 'necessary', 'major', 'primary', 'central', 'main'}
TEMPLATE_PHRASES = {'in this essay', 'in this paper', 'this essay will', 'this paper will', 'this article will', 'the purpose of this', 'this discussion will', 'this article explores', 'it is important to note', 'it is worth noting', 'it should be noted', 'this highlights', 'this demonstrates', 'this suggests', 'this indicates', 'this implies', 'in conclusion', 'to conclude', 'to summarize', 'in summary', 'overall,', 'in closing', 'to sum up'}

DISCOURSE_MARKERS_SET = set(DISCOURSE_MARKERS)
SAFE_VOCAB_SET = set(SAFE_VOCAB)
IMPORTANCE_SET = set(IMPORTANCE_WORDS)

CONTENT_TAGS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}
FUNCTION_TAGS = {'DET', 'PRON', 'ADP', 'AUX', 'CCONJ', 'SCONJ', 'PART'}

INTRO_OUTRO = ["in this essay", "in this paper", "in this article", "this essay will", "this paper will", "this article will", "the purpose of this", "this discussion will", "this article explores", "this paper examines", "this essay discusses", "the aim of this", "the goal of this", "this study explores", "this section discusses", "this text explores", "to begin with", "firstly", "first of all", "on the one hand", "on the other hand", "in addition", "furthermore", "moreover", "additionally", "another important point", "it is important to note", "it should be noted", "this highlights", "this demonstrates", "this suggests", "this indicates", "in conclusion", "to conclude", "to summarize", "in summary", "overall", "ultimately", "in closing", "to sum up", "all in all", "in the end", "taking everything into account", "from the above discussion", "based on the above", "in light of the above"]
INTRO_OUTRO_LOWER = [w.lower() for w in INTRO_OUTRO]

CONTRACTIONS = ["n't", "'re", "'ve", "'ll", "'d", "'m", "won't", "can't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "couldn't", "shouldn't", "wouldn't", "mustn't", "needn't", "mightn't", "shan't", "i'm", "you're", "they're", "we're", "she's", "he's", "it's", "i've", "you've", "they've", "we've", "i'll", "you'll", "they'll", "we'll", "i'd", "you'd", "they'd", "we'd", "he'll", "she'll", "it'll", "he'd", "she'd", "it'd", "who's", "what's", "where's", "when's", "why's", "how's", "ain't", "let's", "there's", "here's", "gonna", "wanna", "gotta", "lemme", "gimme", "kinda", "sorta", "outta", "lotta", "coulda", "shoulda", "woulda", "y'all", "y'all're", "y'all've", "y'all'd"]
INFORMAL_WORDS = ["gonna", "wanna", "gotta", "kinda", "sorta", "yeah", "yep", "nope", "nah", "yup", "ok", "okay", "alright", "hey", "yo", "like", "basically", "actually", "literally", "honestly", "seriously", "just", "really", "so", "well", "anyway", "anyways", "i think", "i guess", "i feel", "i mean", "you know", "sort of", "kind of", "lol", "lmao", "rofl", "omg", "wtf", "idk", "imo", "tbh", "ngl", "smh", "bruh", "bro", "dude", "fam", "btw", "fyi", "asap", "irl", "afaik", "ikr", "nvm", "thx", "pls", "plz", "wow", "ugh", "meh", "oops", "yay", "woah", "whoa", "hmm", "maybe", "probably", "perhaps", "guess", "kind of", "sort of", "super", "really", "very", "so much", "too much", "a lot"]
FIRST_PERSON = ["i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves", "i'm", "i’ve", "i'd", "i’ll", "we're", "we’ve", "we’d", "we’ll", "i think", "i believe", "i feel", "i guess", "i mean", "i suppose", "we think", "we believe", "we feel"]
SECOND_PERSON = ["you", "your", "yours", "yourself", "yourselves", "you're", "you've", "you'd", "you'll", "you know", "you see", "you might", "you could", "you should", "you guys", "y'all", "you all"]

INFORMAL_SET = set(INFORMAL_WORDS)
FIRST_PERSON_SET = set(FIRST_PERSON)
SECOND_PERSON_SET = set(SECOND_PERSON)
CONTRACTIONS_LOWER = [c.lower() for c in CONTRACTIONS]

# ==========================================
# Extraction Functions
# ==========================================
def extract_sentence_features(sentences):
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) < 2:
        return {'sent_len_mean': 0, 'sent_len_std': 0, 'sent_len_min': 0, 'sent_len_max': 0, 'sent_len_range': 0, 'burstiness': 0, 'sent_len_cv': 0}
    mean_l = np.mean(lengths)
    std_l  = np.std(lengths)
    return {'sent_len_mean': mean_l, 'sent_len_std': std_l, 'sent_len_min': np.min(lengths), 'sent_len_max': np.max(lengths), 'sent_len_range': np.max(lengths) - np.min(lengths), 'burstiness': (std_l - mean_l) / (std_l + mean_l + 1e-9), 'sent_len_cv': std_l / (mean_l + 1e-9)}

def extract_lexical_features(words):
    total = len(words)
    if total < 10:
        return {'ttr': 0, 'root_ttr': 0, 'corrected_ttr': 0, 'log_ttr': 0, 'mtld_proxy': 0, 'msttr': 0, 'hapax_ratio': 0, 'dis_ratio': 0, 'rare_ratio': 0, 'freq_entropy': 0, 'gini_coef': 0, 'top10_coverage': 0, 'top1_word_freq': 0, 'vocab_size': 0}
    freq = Counter(words)
    unique = len(freq)
    ttr = unique / total
    root_ttr = unique / np.sqrt(total)
    corrected_ttr = unique / np.sqrt(2 * total)
    log_ttr = np.log(unique + 1) / np.log(total + 1)
    mtld_proxy = total / (unique + 1e-9)
    window = 50
    if total >= window:
        ttrs = []
        for i in range(0, total - window + 1, window // 2):
            chunk = words[i:i+window]
            ttrs.append(len(set(chunk)) / window)
        msttr = np.mean(ttrs)
    else:
        msttr = ttr
    hapax = sum(1 for v in freq.values() if v == 1)
    dis   = sum(1 for v in freq.values() if v == 2)
    freqs = np.array(list(freq.values()))
    probs = freqs / total
    freq_entropy = -np.sum(probs * np.log(probs + 1e-9))
    sorted_probs = np.sort(probs)
    n = len(sorted_probs)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_probs)) / np.sum(sorted_probs)) - (n + 1)
    top10_coverage = sum(v for _, v in freq.most_common(10)) / total
    top1_freq = freq.most_common(1)[0][1] / total
    return {'ttr': ttr, 'root_ttr': root_ttr, 'corrected_ttr': corrected_ttr, 'log_ttr': log_ttr, 'mtld_proxy': mtld_proxy, 'msttr': msttr, 'hapax_ratio': hapax / (unique + 1e-9), 'dis_ratio': dis / (unique + 1e-9), 'rare_ratio': (hapax + dis) / (unique + 1e-9), 'freq_entropy': freq_entropy, 'gini_coef': gini / n, 'top10_coverage': top10_coverage, 'top1_word_freq': top1_freq, 'vocab_size': unique}

def extract_marker_features(text_clean, words):
    tl = text_clean.lower()
    wc = len(words) + 1e-9
    freq = Counter(words)
    sv_count = sum(freq.get(w, 0) for w in SAFE_VOCAB_SET)
    importance_cnt = sum(freq.get(w, 0) for w in IMPORTANCE_SET)
    dm_count = sum(tl.count(p) for p in DISCOURSE_MARKERS)
    template_cnt = sum(tl.count(p) for p in TEMPLATE_PHRASES)
    sentences = sentence_splitter.split(tl)
    sent_marker_count = 0
    sent_total = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent: continue
        sent_total += 1
        if any(' '.join(sent.split()[:4]).startswith(m) for m in DISCOURSE_MARKERS):
            sent_marker_count += 1
    sent_total += 1e-9
    positions = [i for i, w in enumerate(words) if w in IMPORTANCE_SET]
    synonym_cycle = 1 / (np.std(np.diff(positions)) + 1e-6) if len(positions) > 1 else 0
    safe_words = [w for w in words if w in SAFE_VOCAB_SET]
    safe_vocab_div = len(set(safe_words)) / (len(safe_words) + 1e-9)
    stacked_markers = sum(1 for i in range(len(words)-1) if words[i] in DISCOURSE_MARKERS_SET and words[i+1] in DISCOURSE_MARKERS_SET)
    paragraphs = [p.strip() for p in text_clean.split('\n') if p.strip()]
    para_marker_counts = [sum(p.lower().count(m) for m in DISCOURSE_MARKERS) for p in paragraphs]
    marker_burstiness = np.std(para_marker_counts) / (np.mean(para_marker_counts) + 1e-9) if len(para_marker_counts) > 1 else 0
    return {'discourse_marker_rate': dm_count / wc, 'safe_vocab_rate': sv_count / wc, 'importance_word_rate': importance_cnt / wc, 'template_phrase_count': template_cnt, 'it_is_rate': tl.count('it is') / wc, 'this_is_rate': tl.count('this is') / wc, 'there_is_rate': tl.count('there is') / wc, 'sentence_start_marker_rate': sent_marker_count / sent_total, 'synonym_cycle_score': synonym_cycle, 'safe_vocab_diversity': safe_vocab_div, 'intro_template_flag': int(any(p in tl[:200] for p in TEMPLATE_PHRASES)), 'ending_template_flag': int(any(p in tl[-200:] for p in TEMPLATE_PHRASES)), 'stacked_marker_count': stacked_markers, 'marker_burstiness': marker_burstiness}

def normalized_entropy_fast(freq_counts):
    counts = np.array(list(freq_counts.values()), dtype=np.float64)
    total = counts.sum()
    if total == 0 or len(counts) == 0: return 0.0, 0.0
    h = scipy_entropy(counts / total)
    return float(h), float(h / (np.log(len(counts) + 1e-9) + 1e-9))
def repetition_concentration_fast(freq_counts):
    counts = np.array(list(freq_counts.values()), dtype=np.float64)
    return float(np.sum((counts / counts.sum()) ** 2)) if len(counts) > 0 else 0.0
def ngram_stats_fast(words, n):
    return Counter(zip(*[words[i:] for i in range(n)])) if len(words) >= n else Counter()
def extract_entropy_features(words):
    if len(words) < 10:
        return {'word_entropy': 0, 'normalized_word_entropy': 0, 'bigram_entropy': 0, 'normalized_bigram_entropy': 0, 'trigram_entropy': 0, 'normalized_trigram_entropy': 0, 'bigram_reuse_rate': 0, 'trigram_reuse_rate': 0, 'fourgram_reuse_rate': 0, 'word_redundancy': 0, 'long_phrase_redundancy': 0, 'local_entropy_mean': 0, 'local_entropy_std': 0, 'local_entropy_min': 0, 'local_entropy_max': 0, 'entropy_variation_ratio': 0, 'zipf_deviation': 0}
    freq = Counter(words)
    word_entropy, norm_word_entropy = normalized_entropy_fast(freq)
    bg, tg, fg = ngram_stats_fast(words, 2), ngram_stats_fast(words, 3), ngram_stats_fast(words, 4)
    bigram_entropy, norm_bigram_entropy = normalized_entropy_fast(bg)
    trigram_entropy, norm_trigram_entropy = normalized_entropy_fast(tg)
    local_ent = []
    for i in range(len(words) - 30 + 1):
        local_ent.append(scipy_entropy(np.array(list(Counter(words[i:i+30]).values()), dtype=np.float64) / 30))
    local_ent = np.array(local_ent, dtype=np.float64)
    zipf_score = 0.0
    counts = np.array(sorted(freq.values(), reverse=True), dtype=np.float64)
    if len(counts) >= 5: zipf_score = float(abs(np.polyfit(np.log(np.arange(1, len(counts) + 1, dtype=np.float64)), np.log(counts), 1)[0] + 1))
    return {'word_entropy': word_entropy, 'normalized_word_entropy': norm_word_entropy, 'bigram_entropy': bigram_entropy, 'normalized_bigram_entropy': norm_bigram_entropy, 'trigram_entropy': trigram_entropy, 'normalized_trigram_entropy': norm_trigram_entropy, 'bigram_reuse_rate': sum(v>1 for v in bg.values())/(len(bg)+1e-9), 'trigram_reuse_rate': sum(v>1 for v in tg.values())/(len(tg)+1e-9), 'fourgram_reuse_rate': sum(v>1 for v in fg.values())/(len(fg)+1e-9), 'word_redundancy': repetition_concentration_fast(freq), 'long_phrase_redundancy': repetition_concentration_fast(fg), 'local_entropy_mean': float(local_ent.mean()) if len(local_ent)>0 else 0, 'local_entropy_std': float(local_ent.std()) if len(local_ent)>0 else 0, 'local_entropy_min': float(local_ent.min()) if len(local_ent)>0 else 0, 'local_entropy_max': float(local_ent.max()) if len(local_ent)>0 else 0, 'entropy_variation_ratio': (float(local_ent.std()) / (float(local_ent.mean()) + 1e-9)) if len(local_ent)>0 else 0, 'zipf_deviation': zipf_score}

def fit_zipf_segment(log_r, log_c):
    if len(log_r) < 5 or np.all(log_c == log_c[0]): return 0.0, 0.0, 0.0
    coeffs = np.polyfit(log_r, log_c, 1)
    return float(coeffs[0]), float(spearmanr(log_r, log_c)[0]**2) if not np.isnan(spearmanr(log_r, log_c)[0]) else 0.0, float(np.std(log_c - np.polyval(coeffs, log_r)))
def extract_zipf_features(words):
    if len(words) < 50: return {'zipf_slope': 0, 'zipf_r_squared': 0, 'zipf_residual_std': 0, 'zipf_residual_mean_abs': 0, 'zipf_slope_deviation': 0, 'zipf_head_slope': 0, 'zipf_tail_slope': 0, 'zipf_head_r2': 0, 'zipf_tail_r2': 0, 'zipf_head_tail_gap': 0, 'zipf_curvature': 0, 'zipf_tail_smoothness': 0, 'hapax_ratio': 0, 'dis_ratio': 0, 'top_10_token_mass': 0, 'top_50_token_mass': 0, 'vocab_token_ratio': 0, 'zipf_constant_flag': 0}
    freq = Counter(words)
    counts = np.array(sorted(freq.values(), reverse=True), dtype=np.float64)
    log_r, log_c = np.log(np.arange(1, len(counts) + 1, dtype=np.float64)), np.log(counts)
    slope = float(np.polyfit(log_r, log_c, 1)[0])
    residuals = log_c - np.polyval(np.polyfit(log_r, log_c, 1), log_r)
    split = max(10, len(log_r) // 5)
    hs, hr, _ = fit_zipf_segment(log_r[:split], log_c[:split])
    ts, tr, _ = fit_zipf_segment(log_r[split:], log_c[split:])
    return {'zipf_slope': slope, 'zipf_r_squared': float(spearmanr(log_r,log_c)[0]**2) if not np.isnan(spearmanr(log_r,log_c)[0]) and not np.all(log_c==log_c[0]) else 0.0, 'zipf_residual_std': float(np.std(residuals)), 'zipf_residual_mean_abs': float(np.mean(np.abs(residuals))), 'zipf_slope_deviation': float(abs(slope + 1)), 'zipf_head_slope': hs, 'zipf_tail_slope': ts, 'zipf_head_r2': hr, 'zipf_tail_r2': tr, 'zipf_head_tail_gap': float(abs(hs - ts)), 'zipf_curvature': float(np.polyfit(log_r, log_c, 2)[0]), 'zipf_tail_smoothness': float(np.std(np.diff(np.log(counts[split:])))) if len(counts[split:]) > 2 else 0.0, 'hapax_ratio': float(np.sum(counts == 1) / len(counts)), 'dis_ratio': float(np.sum(counts == 2) / len(counts)), 'top_10_token_mass': float(np.sum(counts[:10]) / counts.sum()), 'top_50_token_mass': float(np.sum(counts[:50]) / counts.sum()), 'vocab_token_ratio': float(len(freq) / len(words)), 'zipf_constant_flag': int(np.all(counts == counts[0]))}

def transition_smoothness(tags):
    if len(tags) < 2: return 0.0
    probs = np.array(list(Counter(zip(tags[:-1], tags[1:])).values()), dtype=np.float64)
    return float(np.sum((probs / probs.sum()) ** 2))
def pos_features_from_doc(doc):
    tags = [t.pos_ for t in doc if not t.is_space]
    total = len(tags) + 1e-9
    tc = Counter(tags)
    sent_len, noun_rate, verb_rate = [], [], []
    for sent in doc.sents:
        t = [w.pos_ for w in sent if not w.is_space]
        if t: sent_len.append(len(t)); noun_rate.append(Counter(t).get('NOUN',0)/len(t)); verb_rate.append(Counter(t).get('VERB',0)/len(t))
    content_rate = sum(tc.get(t, 0) for t in CONTENT_TAGS) / total
    function_rate = sum(tc.get(t, 0) for t in FUNCTION_TAGS) / total
    return {'pos_noun_rate': tc.get('NOUN', 0) / total, 'pos_verb_rate': tc.get('VERB', 0) / total, 'pos_adj_rate': tc.get('ADJ', 0) / total, 'pos_adv_rate': tc.get('ADV', 0) / total, 'pos_punct_rate': tc.get('PUNCT', 0) / total, 'conj_rate': (tc.get('CCONJ', 0) + tc.get('SCONJ', 0)) / total, 'aux_rate': tc.get('AUX', 0) / total, 'noun_verb_ratio': tc.get('NOUN', 0) / (tc.get('VERB', 0) + 1e-9), 'content_function_ratio': content_rate / (function_rate + 1e-9), 'content_function_diff': content_rate - function_rate, 'pos_entropy': float(scipy_entropy(np.array(list(tc.values())) / total)) if tags else 0.0, 'pos_bigram_reuse': sum(v>1 for v in Counter(zip(*(tags[i:] for i in range(2)))).values()) / (len(tags) + 1e-9) if len(tags) >= 2 else 0.0, 'pos_trigram_reuse': sum(v>1 for v in Counter(zip(*(tags[i:] for i in range(3)))).values()) / (len(tags) + 1e-9) if len(tags) >= 3 else 0.0, 'transition_concentration': transition_smoothness(tags), 'sentence_length_std_pos': float(np.std(sent_len)) if sent_len else 0.0, 'noun_rate_sentence_std': float(np.std(noun_rate)) if noun_rate else 0.0, 'verb_rate_sentence_std': float(np.std(verb_rate)) if verb_rate else 0.0, 'modifier_stack_rate': sum(1 for i in range(len(tags)-1) if tags[i] in {'ADJ','ADV'} and tags[i+1] in {'ADJ','ADV'}) / total}

def extract_informality_features_fast(text_clean, words, spell):
    tl = text_clean.lower()
    wc = len(words) + 1e-9
    freq = Counter(words)
    sample_words = [w for w in words[:200] if len(w) > 2]
    return {'typo_rate': len(spell.unknown(sample_words)) / (len(sample_words) + 1e-9), 'contraction_rate': sum(tl.count(c) for c in CONTRACTIONS_LOWER) / wc, 'informal_rate': sum(freq.get(w, 0) for w in INFORMAL_SET) / wc, 'first_person_rate': sum(freq.get(w, 0) for w in FIRST_PERSON_SET) / wc, 'second_person_rate': sum(freq.get(w, 0) for w in SECOND_PERSON_SET) / wc, 'question_rate': text_clean.count('?') / wc, 'exclamation_rate': text_clean.count('!') / wc, 'em_dash_rate': (text_clean.count('—') + text_clean.count('--')) / wc, 'ellipsis_rate': text_clean.count('...') / wc}

def extract_structural_features_fast(text):
    paras = [p.strip() for p in text.split('\n') if len(p.strip()) > 20]
    para_lens = [len(p.split()) for p in paras]
    pmean = float(np.mean(para_lens)) if len(para_lens) >= 2 else float(len(text.split()))
    pstd = float(np.std(para_lens)) if len(para_lens) >= 2 else 0.0
    lines = [l for l in text.split('\n') if l.strip()]
    llens = [len(l.split()) for l in lines]
    list_lines = sum(1 for ls in (l.strip() for l in lines) if ls and (ls[0].isdigit() or ls.startswith('-') or ls.startswith('*') or ls.startswith('•')))
    bullet_styles = [('num' if ls[0].isdigit() else ls[0]) for ls in (l.strip() for l in lines) if ls and (ls[0].isdigit() or ls.startswith('-') or ls.startswith('*') or ls.startswith('•'))]
    return {'para_count': len(para_lens), 'para_len_mean': pmean, 'para_len_std': pstd, 'para_uniformity': pstd / (pmean + 1e-9), 'para_min_len': int(np.min(para_lens)) if len(para_lens) >= 2 else int(pmean), 'para_max_len': int(np.max(para_lens)) if len(para_lens) >= 2 else int(pmean), 'para_range': (int(np.max(para_lens)) - int(np.min(para_lens))) if len(para_lens) >= 2 else 0, 'para_progression_slope': float(np.polyfit(np.arange(len(para_lens), dtype=np.float64), para_lens, 1)[0]) if len(para_lens) >= 3 else 0.0, 'list_rate': list_lines / (len(lines) + 1e-9), 'list_style_consistency': len(set(bullet_styles)) / (len(bullet_styles) + 1e-9) if bullet_styles else 0.0, 'line_len_mean': float(np.mean(llens)) if llens else 0.0, 'line_len_std': float(np.std(llens)) if llens else 0.0, 'line_uniformity': float(np.std(llens)) / (float(np.mean(llens)) + 1e-9) if llens else 0.0, 'template_count': sum(text.lower().count(w) for w in INTRO_OUTRO_LOWER), 'header_count': sum(1 for ls in (l.strip().lower() for l in lines) if header_pattern.match(ls) or (ls.endswith(':') and len(ls.split()) <= 6)), 'intro_outro_balance': float(abs(para_lens[0] - para_lens[-1])) if len(para_lens) >= 2 else 0.0}

def extract_sbert_features(sbert, sentences):
    if len(sentences) < 2: return {'sbert_adj_sim_mean': 0.0, 'sbert_adj_sim_std': 0.0, 'sbert_adj_sim_max': 0.0, 'sbert_global_sim_mean': 0.0, 'sbert_sim_variance': 0.0, 'sbert_coherence_drop': 0.0}
    embs = sbert.encode(sentences[:20], batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
    adj = np.sum(embs[:-1] * embs[1:], axis=1)
    if len(embs) <= 15:
        upper = (embs @ embs.T)[np.triu_indices(len(embs), k=1)]
    else:
        sampled = embs[np.random.choice(len(embs), 15, replace=False)]
        upper = (sampled @ sampled.T)[np.triu_indices(15, k=1)]
    return {'sbert_adj_sim_mean': float(adj.mean()), 'sbert_adj_sim_std': float(adj.std()), 'sbert_adj_sim_max': float(adj.max()), 'sbert_global_sim_mean': float(upper.mean()), 'sbert_sim_variance': float(upper.var()), 'sbert_coherence_drop': float(adj[0] - adj[-1]) if len(adj) > 1 else 0.0}

# ==========================================
# Main Compiler Logic natively requested by Server 
# ==========================================
def extract_all_features_native(text, nlp, sbert, spell):
    if not isinstance(text, str): text = ""
    text_clean = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', text.encode('utf-8', errors='ignore').decode('utf-8'))).strip()
    if text_clean.startswith('"') and text_clean.endswith('"'): text_clean = text_clean[1:-1].strip()
    
    doc = nlp(text_clean)
    sents = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 3]
    words = [t.lower() for t in word_pattern.findall(doc.text)]
    
    feat = {}
    feat.update(extract_sentence_features(sents))
    feat.update(extract_lexical_features(words))
    feat.update(extract_marker_features(text_clean, words))
    feat.update(extract_entropy_features(words))
    feat.update(extract_zipf_features(words))
    feat.update(pos_features_from_doc(doc))
    feat.update(extract_informality_features_fast(text_clean, words, spell))
    feat.update(extract_structural_features_fast(text_clean))
    feat.update(extract_sbert_features(sbert, sents))
    
    return feat
