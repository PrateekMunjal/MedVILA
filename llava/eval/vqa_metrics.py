import json
import torch
import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import sentence_rouge
from sentence_transformers import SentenceTransformer, util

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load BERT model for semantic similarity
bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load JSON file
with open("slake_predictions_testset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Define metric functions

## 1. Exact Match Accuracy
def exact_match(pred, gt):
    return int(pred.strip().lower() == gt.strip().lower())

## 2. BLEU Score
def compute_bleu(pred, gt):
    reference = [gt.split()]  # Ground truth as reference
    candidate = pred.split()  # Predicted response
    return sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)

## 3. ROUGE-L Score
def compute_rouge(pred, gt):
    rouge_scores = sentence_rouge.compute_rouge([pred], [[gt]])
    return rouge_scores["rougeL"]

## 4. METEOR Score (Using nltk's wordnet-based approach)
def compute_meteor(pred, gt):
    return nltk.translate.meteor_score.meteor_score([gt.split()], pred.split())

## 5. WordNet-based Semantic Similarity (Using BERT)
def compute_semantic_similarity(pred, gt):
    pred_embedding = bert_model.encode(pred, convert_to_tensor=True)
    gt_embedding = bert_model.encode(gt, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(pred_embedding, gt_embedding).item()
    return similarity_score

# Initialize metric accumulators
total_exact_match = 0
total_bleu = 0
total_rouge = 0
total_meteor = 0
total_semantic_similarity = 0

num_samples = len(data)

# Compute metrics for each example
for entry in data:
    pred = entry["predicted_response"]
    gt = entry["ground_truth"]
    
    total_exact_match += exact_match(pred, gt)
    total_bleu += compute_bleu(pred, gt)
    total_rouge += compute_rouge(pred, gt)
    total_meteor += compute_meteor(pred, gt)
    total_semantic_similarity += compute_semantic_similarity(pred, gt)

# Calculate final averages
final_metrics = {
    "Exact Match Accuracy": total_exact_match / num_samples,
    "BLEU Score": total_bleu / num_samples,
    "ROUGE-L Score": total_rouge / num_samples,
    "METEOR Score": total_meteor / num_samples,
    "Semantic Similarity (BERT)": total_semantic_similarity / num_samples,
}

# Save results
with open("vqa_metrics_results.json", "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, indent=4)

# Print results
print("VQA Metrics Calculation Completed:")
print(json.dumps(final_metrics, indent=4))
