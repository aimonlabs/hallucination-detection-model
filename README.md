# Hallucination Detection Model (HDM-2)

AIMon's Hallucination Detection Model-2 (HDM-2) is a powerful tool for identifying hallucinations in large language model (LLM) responses. This repository contains the inference code for HDM-2, allowing developers to integrate hallucination detection into their AI pipelines.

## Features

- **Token-level Detection**: Identifies specific hallucinated words and spans
- **Sentence-level Classification**: Classifies entire sentences as hallucinated or factual
- **Severity Scoring**: Provides a quantitative measure of hallucination severity
- **Flexible Integration**: Easy to integrate with existing LLM applications
- **Optimized Performance**: Supports both CPU and GPU inference with optional quantization

## Installation

### From PyPI (Recommended)

```bash
pip install hdm2
```

### From Source

```bash
git clone https://github.com/aimonlabs/hallucination-detection-model.git
cd hallucination-detection-model
pip install -e .
```

For GPU acceleration (recommended for production use):

```bash
pip install hdm2[gpu]
```

## Quick Start

```python
from hdm2 import HallucinationDetectionModel

# Initialize the model
hdm = HallucinationDetectionModel()

# Prepare your inputs
prompt = "Explain quantum computing"
context = "Quantum computing is a type of computing that uses quantum phenomena such as superposition and entanglement. A quantum computer maintains a sequence of qubits."
response = "Quantum computing is a revolutionary field that leverages the principles of quantum mechanics to perform computations. Unlike classical computing which uses bits, quantum computing uses quantum bits or qubits. These qubits can exist in multiple states simultaneously due to a phenomenon called superposition. Additionally, quantum entanglement allows qubits to be intrinsically connected regardless of distance. Quantum computers are particularly effective for solving complex problems like factoring large numbers and simulating quantum systems. They are also good at optimization problems and machine learning tasks."

# Detect hallucinations
results = hdm.apply(prompt, context, response)

# Check results
if results['hallucination_detected']:
    print(f"Hallucination detected with severity: {results['hallucination_severity']:.4f}")
    
    # Print hallucinated sentences
    print("\nHallucinated sentences:")
    for sentence_result in results['ck_results']:
        if sentence_result['prediction'] == 1:  # 1 indicates hallucination
            print(f"- {sentence_result['text']}")
else:
    print("No hallucinations detected.")
```

## Advanced Usage

### Customizing Detection Parameters

```python
# Initialize with custom device and quantization options
hdm = HallucinationDetectionModel(
    device="cuda",  # Force CUDA (GPU) usage
    load_in_8bit=True  # Use 8-bit quantization to reduce memory usage
)

# Customize detection thresholds and options
results = hdm.apply(
    prompt=prompt,
    context=context, 
    response=response,
    token_threshold=0.6,  # Increase token-level threshold (0-1)
    ck_threshold=0.8,     # Increase sentence-level threshold (0-1)
    debug=True            # Enable debug output
)
```

### Loading from Local Path

If you've previously downloaded the model:

```python
hdm = HallucinationDetectionModel(
    model_components_path="path/to/model_components/",
    ck_classifier_path="path/to/ck_classifier/"
)
```

## Output Format

The `apply()` method returns a dictionary with the following keys:

- `hallucination_detected` (bool): Whether any hallucination was detected
- `hallucination_severity` (float): Overall hallucination severity score (0-1)
- `ck_results` (list): Per-sentence results with hallucination probabilities
- `high_scoring_words` (list): Words/spans with high hallucination scores
- `candidate_sentences` (list): Sentences with potential hallucinations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the license included in the repository.

## Citation

If you use HDM-2 in your research, please cite:

```
@misc{paudel2025hallucinothallucinationdetectioncontext,
      title={HalluciNot: Hallucination Detection Through Context and Common Knowledge Verification}, 
      author={Bibek Paudel and Alexander Lyzhov and Preetam Joshi and Puneet Anand},
      year={2025},
      eprint={2504.07069},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.07069}, 
}
```
