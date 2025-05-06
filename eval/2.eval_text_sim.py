import json
import spacy
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from tqdm import tqdm

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt')
nltk.download('punkt_tab')

nlp = spacy.load("en_core_web_sm")

rouge = Rouge()
smooth_fn = SmoothingFunction().method1

def extract_entities(text):
    doc = nlp(text)
    return set(ent.text.strip() for ent in doc.ents)

def compute_bleu(reference, generated):
    reference_tokens = nltk.word_tokenize(reference)
    generated_tokens = nltk.word_tokenize(generated)
    return sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smooth_fn)

def compute_rouge(reference, generated):
    if not generated.strip():
        return {
            "rouge-1": {"f": 0.0},
            "rouge-2": {"f": 0.0},
            "rouge-l": {"f": 0.0}
        }
    return rouge.get_scores(generated, reference)[0]

def compute_meteor(reference, generated):
    reference_tokens = nltk.word_tokenize(reference)
    generated_tokens = nltk.word_tokenize(generated)
    return meteor_score([reference_tokens], generated_tokens)

def compute_cosine(reference, generated):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([reference, generated])
    cos_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return cos_sim

def compute_ner_overlap(input_text, reference, generated):
    entities_input = extract_entities(input_text)
    entities_reference = extract_entities(reference)
    entities_generated = extract_entities(generated)

    new_entities_ref = entities_reference - entities_input
    new_entities_gen = entities_generated - entities_input

    overlap = new_entities_ref.intersection(new_entities_gen)

    ratio = len(overlap) / len(new_entities_ref) if len(new_entities_ref) > 0 else 0.0

    return {
        "overlap_entities": list(overlap),
        "ner_overlap_ratio": ratio
    }

def main(input_file, output_file):
    results = []

    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing records"):
            data = json.loads(line)
            input_text = data['input']
            generated_output = data['generated_output']
            reference_output = data['reference_output']

            bleu = compute_bleu(reference_output, generated_output)
            rouge_scores = compute_rouge(reference_output, generated_output)
            meteor = compute_meteor(reference_output, generated_output)
            cosine = compute_cosine(reference_output, generated_output)
            ner_metrics = compute_ner_overlap(input_text, reference_output, generated_output)

            result = {
                **data,
                "metrics": {
                    "bleu": bleu,
                    "rouge-1": rouge_scores['rouge-1']['f'],
                    "rouge-2": rouge_scores['rouge-2']['f'],
                    "rouge-l": rouge_scores['rouge-l']['f'],
                    "meteor": meteor,
                    "cosine_similarity": cosine,
                    "ner_overlap_ratio": ner_metrics['ner_overlap_ratio'],
                    "overlap_entities": ner_metrics['overlap_entities']
                }
            }
            results.append(result)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(results, out_f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    input_path = "exp_dense1gpu_batch_8.jsonl"
    output_path = "exp_dense1gpu_batch_8.json"
    main(input_path, output_path)