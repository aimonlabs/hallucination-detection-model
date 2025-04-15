# Hallucination Detection Model (HDM-2)

<p align="center">
  <img src="images/aimon_logo.png" alt="AIMon Logo">
</p>

AIMon's Hallucination Detection Model-2 (HDM-2) is a powerful tool for identifying hallucinations in large language model (LLM) responses. This repository contains the inference code for HDM-2, allowing developers to integrate hallucination detection into their AI pipelines.

## Features

<p align="center">
  <img src="images/taxonomy.png" alt="LLM Response Taxonomy">
</p>

As shown in the figure above, an LLM response can be broken down into context based generation, common knowledge based generation, enterprise knowledge based generation and innocuous statments. 

HDM-2 offers the following features to help classify the output into this taxonomy.

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
prompt = "Describe what penguins are"
context = """
Penguins are flightless aquatic birds that live almost exclusively in the Southern Hemisphere. They are highly adapted for life in the water, with a countershaded dark and white plumage.
"""
response = """
Penguins are flightless aquatic birds that have evolved to thrive in cold environments, primarily in the Southern Hemisphere. Their bodies are perfectly adapted for marine life - they have wings that have evolved into flippers for swimming, dense waterproof feathers for insulation, and a countershaded dark and white plumage that provides camouflage while swimming. The black back and white front coloration helps them blend in when viewed from above or below in the water. Penguins feed primarily on fish, squid, and krill, which they catch while swimming underwater. They are highly social birds that nest in colonies, sometimes containing thousands of individuals. Of the 18 penguin species, the Emperor penguin is the largest, standing about 1.1 meters tall, while the Little Blue penguin is the smallest at around 40 centimeters.
"""

# Detect hallucinations
results = hdm.apply(prompt, context, response)

# Check results
if results['hallucination_detected']:
    print(f"Hallucination detected with severity: {results['adjusted_hallucination_severity']:.4f}")
    
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
- `hallucination_severity` (float): Overall hallucination severity score (0.0-1.0)
- `adjusted_hallucination_severity`(float): Adjusted hallucination severity score (0.0-1.0) that incorporates the results from the common knowledge model. It's value is 0.0 if all candidate sentences are common knowledge.
- `ck_results` (list): Per-sentence results with hallucination probabilities
- `high_scoring_words` (list): Words/spans with high hallucination scores
- `candidate_sentences` (list): Sentences with potential hallucinations

## Model Weights and Evaluation Dataset on HuggingFace 🤗

As a service to the community, we are releasing the weights for our 3B parameter model, along with the evaluation split of our dataset HDMBench.
Please refer to the paper (linked below) for details on the dataset and the model architecture.

Note that this dataset is meant only for benchmarking, and it should not be used for training or hyperparameter-tuning.

Model weights on HF [here](https://huggingface.co/AimonLabs/hallucination-detection-model/).

HDMBench evaluation split on HF [here](https://huggingface.co/datasets/AimonLabs/HDM-Bench).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the license included here:

CC BY-NC-SA 4.0

License URL: https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1

## Citation

The full-text of our paper 📃 is available on arXiv [here](https://arxiv.org/abs/2504.07069).

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
