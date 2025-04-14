import logging
logging.basicConfig(level=logging.INFO)

from hdm2.model import HallucinationDetectionModel

def main():
    # Initialize the model
    # Option 1: Using HuggingFace (default)
    # Replace "your-repo-id" with the actual HF repository ID
    hdm = HallucinationDetectionModel(repo_id="your-repo-id")
    
    # Option 2: Using local paths
    # hdm = HallucinationDetectionModel(
    #     model_components_path="../models/token_seq_model/model_components/",
    #     ck_classifier_path="ck_classifier_op_2/checkpoint-4802/"
    # )
    
    # Example inputs
    prompt = "Explain quantum computing"
    context = "Quantum computing is a type of computing that uses quantum phenomena such as superposition and entanglement. A quantum computer maintains a sequence of qubits."
    response = "Quantum computing is a revolutionary field that leverages the principles of quantum mechanics to perform computations. Unlike classical computing which uses bits, quantum computing uses quantum bits or qubits. These qubits can exist in multiple states simultaneously due to a phenomenon called superposition. Additionally, quantum entanglement allows qubits to be intrinsically connected regardless of distance. Quantum computers are particularly effective for solving complex problems like factoring large numbers and simulating quantum systems. They are also good at optimization problems and machine learning tasks. Currently, companies like IBM, Google, and Microsoft are developing quantum computers with increasing numbers of qubits."
    
    # Detect hallucinations with default parameters
    results = hdm.apply(prompt, context, response)
    
    # Print results
    print(f"Hallucination detected: {results['hallucination_detected']}")
    print(f"Hallucination severity: {results['hallucination_severity']:.4f}")
    
    # Print hallucinated sentences
    if results['candidate_sentences']:
        print("\nPotentially hallucinated sentences:")
        for sentence_result in results['ck_results']:
            if sentence_result['prediction'] == 1:  # 1 indicates hallucination
                print(f"- {sentence_result['text']} (Probability: {sentence_result['hallucination_probability']:.4f})")
    else:
        print("\nNo hallucinated sentences detected.")
    
    # Access high-scoring word spans if available
    if results['high_scoring_words']:
        print("\nHigh-scoring words/spans:")
        for word_info in results['high_scoring_words']:
            if 'word' in word_info:
                print(f"- {word_info['word']} (Score: {word_info['score']:.4f})")
            else:
                print(f"- Span at positions {word_info['start']}-{word_info['end']} (Score: {word_info['score']:.4f})")

if __name__ == "__main__":
    main() 