# 073 — Deep Contextualized Word Representations (ELMo)

**Paper:** Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). *Deep Contextualized Word Representations*. NAACL 2018.

**arXiv:** https://arxiv.org/abs/1802.05365

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/073-elmo-deep-contextualized-word-representations)

---

## Summary

ELMo (Embeddings from Language Models) introduced a new type of deep contextualized word representation that models both complex characteristics of word use (syntax and semantics) and how these uses vary across linguistic contexts (i.e., polysemy). Instead of assigning each word a single static vector (as Word2Vec or GloVe do), ELMo representations are learned functions of the internal states of a deep bidirectional language model (biLM) pre-trained on a large text corpus. The biLM consists of multiple layers of bidirectional LSTM cells. For each token, ELMo concatenates the hidden states from the forward and backward LSTMs at each layer. The final representation is a weighted sum of all layers, where the layer weights are learned task-specifically during downstream fine-tuning.

The key insight is that different layers of the biLM capture different aspects of language: lower layers encode syntax (part-of-speech, morphology), while higher layers encode semantics (word sense, coreference). By exposing all layers and letting downstream models learn how to combine them, ELMo achieves state-of-the-art results across six NLP benchmarks. The paper showed that adding ELMo to existing models — from BiLSTM taggers to attention-based QA systems — yields consistent improvements of 4–25% relative error reduction without architecture changes.

### Core Method Details

1. **Bidirectional Language Model (biLM):** A multi-layer forward LSTM predicts the next token given previous tokens; a backward LSTM predicts the previous token given future tokens. Both share a character-level CNN token encoder (though our implementation uses a simpler word-level vocabulary for clarity).
2. **Layer-wise Embeddings:** At each LSTM layer *j*, the forward hidden state *h_j^fwd* and backward hidden state *h_j^bwd* are concatenated to form the layer-j contextual embedding. The input embedding (layer 0) is the non-contextual token representation.
3. **Learnable Layer Weights:** The final ELMo vector is a task-specific weighted combination: *ELMo = γ · Σ_j s_j · h_j*, where *s_j* are softmax-normalized learnable scalars and *γ* is a scaling factor. These weights are learned during downstream fine-tuning.
4. **Context Sensitivity:** Because the biLM processes the entire sentence, the same word type in different sentences (or different positions within the same sentence) produces different ELMo vectors — directly modeling polysemy.

### Influence

With over 12,000 citations, ELMo was a landmark paper that bridged the static-embedding era (Word2Vec, GloVe) and the fine-tuned-pretrained-model era (BERT, GPT). It demonstrated that pre-trained contextual representations could be "plugged in" to virtually any NLP model, a paradigm that BERT would popularize later the same year. ELMo won the Best Paper Award at NAACL 2018.

---

## What problem does it solve

Imagine you have a dictionary where the word "bank" has only one definition. But in real life, "bank" can mean the side of a river ("I sat by the bank of the river") or a place that keeps money ("I deposited cash at the bank"). Old word-embedding systems like Word2Vec gave "bank" the same vector no matter which meaning you intended — it was like a dictionary with only one entry per word.

ELMo fixes this by looking at the whole sentence before deciding what "bank" means. It's like a smart friend who knows that if you mention "river" and "sitting," you probably mean the riverbank, and if you mention "deposit" and "cash," you mean the money bank. ELMo reads the entire sentence, understands the context, and gives "bank" a different vector in each sentence. This means computers can finally tell apart words that are spelled the same but mean different things — just like humans do.
