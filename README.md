# Generative AI Image Watermarking and Verification

### Efficient Prompt Embedding Prediction with Watermark Detection and Image Integrity Verification

This repository contains an MSc dissertation project investigating efficient and robust watermarking and verification for generative AI images.

The main research contribution is a **Residual Prompt Embedding Predictor** that approximates an expensive iterative prompt-embedding optimisation stage using a single neural-network inference pass. The project evaluates whether this efficiency improvement can maintain comparable watermark robustness and image quality under progressively stronger image manipulations.

The implementation is built on the **ROBIN watermarking framework** by Huang, Wu and Wang (NeurIPS 2024), which is used as the baseline and technical foundation for the diffusion watermarking pipeline.

The repository also includes a Flask-based research prototype for image generation and verification. The application extends the verification workflow with a separate **Image Signature** mechanism that provides source information and checks whether a signed image has been modified.

---

## Project Overview

The original ROBIN approach obtains an optimised conditioning embedding through iterative adversarial optimisation.

```text
Text Prompt
    |
    v
CLIP Text Encoder
    |
    v
Original Prompt Embedding
    |
    v
Iterative ROBIN Optimisation
    |
    v
Optimised Embedding
    |
    v
Watermarked Image
```

This project investigates replacing the iterative prompt-embedding optimisation stage with a learned Predictor:

```text
Text Prompt
    |
    v
CLIP Text Encoder
    |
    v
Original Prompt Embedding
    |
    v
Residual Predictor
    |
    v
Predicted Optimised Embedding
    |
    v
Watermarked Image
```

The objective is not to replace ROBIN's underlying watermarking mechanism. Instead, the project investigates whether the computational cost of obtaining the optimised prompt embedding can be reduced while preserving comparable watermark robustness and perceptual image quality.

---

## Main Contributions

### 1. Residual Prompt Embedding Predictor

A lightweight residual neural network is trained to approximate the mapping:

```text
Original CLIP Embedding -> ROBIN Optimised Embedding
```

Rather than predicting an entirely new representation, the network learns a correction to the original embedding:

```text
Prediction = Input + MLP(Input)
```

Conceptually:

```text
Original CLIP Embedding
        |
        v
      Linear
        |
        v
       GELU
        |
        v
     Dropout
        |
        v
      Linear
        |
        v
Learned Residual Correction
        |
        + <---- Original Embedding
        |
        v
Predicted Optimised Embedding
```

The Predictor therefore attempts to approximate the result of ROBIN's iterative prompt-embedding optimisation using a single neural-network inference pass.

---

### 2. ROBIN vs Predictor Evaluation

The evaluation pipeline directly compares:

- Original ROBIN
- ROBIN with the learned Residual Predictor

The underlying Stable Diffusion, watermarking, attack and verification mechanisms are kept consistent between the two approaches.

The principal experimental difference is how the optimised conditioning embedding is obtained:

```text
Original ROBIN
      |
      v
Iterative Optimisation
      |
      v
Optimised Embedding
```

versus:

```text
Proposed Method
      |
      v
Residual Predictor
      |
      v
Predicted Optimised Embedding
```

This enables a direct comparison of efficiency, watermark robustness and perceptual image quality.

---

### 3. Progressive Content-Laundering Attacks

The evaluation includes progressively stronger image manipulations designed to test the robustness of watermark verification.

Attack families include:

- JPEG compression
- Cropping
- Gaussian blur
- Rotation
- Gaussian noise
- Colour jitter
- Selected combined attacks

Multiple severity levels are evaluated for each individual attack family to investigate where watermark detection performance begins to degrade.

Evaluation metrics include:

- AUC
- Accuracy
- TPR at 1% FPR
- PSNR
- SSIM
- MS-SSIM

---

### 4. Evaluation on 70 Unseen Prompts

The proposed method was evaluated using two separate sets of prompts that were not used for Predictor training:

- an initial set of **20 unseen prompts**;
- an additional set of **50 unseen prompts**.

The prompt sets were checked for overlap and contain different prompts, giving a total of **70 unseen prompts used for evaluation**.

The final 50-prompt attack sweep produced the following aggregate results across the tested individual attack settings:

| Metric | Original ROBIN | Predictor |
|---|---:|---:|
| Mean AUC | 0.9636 | 0.9648 |
| Mean Accuracy | 0.9367 | 0.9383 |
| Mean TPR @ 1% FPR | 0.7867 | 0.8083 |
| PSNR | 20.0568 | 20.0506 |
| SSIM | 0.6129 | 0.6132 |
| MS-SSIM | 0.7729 | 0.7734 |

The results indicate comparable overall watermark robustness and perceptual quality between baseline ROBIN and the Predictor in the final 50-prompt evaluation.

The values in the table above refer specifically to the **final 50-prompt sweep** and are not aggregate statistics calculated across all 70 prompts.

---

### 5. Image Signature and Integrity Checking

The web application contains an additional **Image Signature** mechanism that operates separately from ROBIN watermark detection.

Generated images can contain signature information including:

- Image identifier
- Creator identifier
- Generation method
- Creation time
- Image-content integrity information

During verification, the application performs two independent checks.

#### Watermark Check

Determines whether the supported ROBIN watermark is detectable in the uploaded image.

#### Image Signature Check

Checks the Image Signature and determines whether the signed image content has subsequently been modified.

Conceptually:

```text
                    Uploaded Image
                         |
              +----------+----------+
              |                     |
              v                     v
       Watermark Check       Image Signature Check
              |                     |
              v                     v
       ROBIN watermark        Signature validity
          detection           + image integrity
```

The Image Signature functionality is an additional prototype feature developed for this project and should not be interpreted as part of the original ROBIN framework.

---

## Repository Structure

Important project components include:

```text
ROBIN/
|
|-- gen_clean_image.py
|-- gen_watermark.py
|-- stable_diffusion_robin.py
|-- inverse_stable_diffusion.py
|-- optim_utils.py
|-- io_utils.py
|
|-- build_training_dataset.py
|-- train_predictor.py
|-- predict_acond.py
|-- test_predictor.py
|
|-- inject_wm_baseline.py
|-- inject_wm_predictor.py
|-- inject_wm_with_predictor.py
|
|-- benchmark_predictor.py
|-- benchmark_opt_vs_predictor.py
|
|-- check_prompt_leakage.py
|-- check_new_prompt_overlap.py
|
|-- run_attack_sweep.py
|-- run_attack_sweep_20_and_100.ps1
|-- run_attack_sweep_50.ps1
|
|-- new_unseen_prompts_20.txt
|-- new_unseen_prompts_50.txt
|
`-- webapp/
    |-- app.py
    |-- generate_signature_keys.py
    |-- test_image_signature.py
    |-- test_signature_tampering.py
    |
    |-- services/
    |   |-- generator.py
    |   |-- verifier.py
    |   `-- image_signature.py
    |
    |-- templates/
    |   |-- base.html
    |   |-- index.html
    |   |-- generate.html
    |   |-- result.html
    |   |-- verify.html
    |   `-- about.html
    |
    `-- static/
        `-- style.css
```

Some original ROBIN files remain in the repository because the proposed method is implemented as an extension of the original ROBIN pipeline.

---

## Software Environment

The project was developed using the following working environment:

- Python 3.10.11
- PyTorch 1.13.1+cu117
- torchvision 0.14.1+cu117
- transformers 4.30.2
- diffusers 0.11.1
- datasets 5.0.0
- accelerate 0.12.0
- scikit-learn 1.7.2
- scikit-image 0.25.2
- Flask 3.1.3

A CUDA-enabled NVIDIA GPU is strongly recommended for Stable Diffusion generation and ROBIN watermark verification.

PyTorch and torchvision were installed separately with CUDA 11.7 support:

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

The remaining dependencies can be installed using:

```bash
pip install -r requirements.txt
```

The experiments use:

```text
runwayml/stable-diffusion-v1-5
```

> **Compatibility note:** The ROBIN inversion pipeline relies on older versions of Diffusers. Newer versions may not be directly compatible with the implementation.

---

## Predictor Workflow

The Predictor pipeline consists of three main stages.

### 1. Generate ROBIN Training Pairs

ROBIN optimisation outputs are used to construct training pairs containing the original prompt embedding and corresponding optimised embedding.

Conceptually:

```text
cond_embedding -> opt_acond
```

These pairs provide the supervised training data for the Residual Predictor.

---

### 2. Train the Predictor

The Predictor can be trained using:

```bash
python train_predictor.py
```

The trained Predictor checkpoint is stored locally and is intentionally excluded from version control.

---

### 3. Predictor Inference

Predictor inference can be tested using:

```bash
python predict_acond.py
```

The Predictor estimates the optimised embedding directly from the original CLIP prompt embedding.

---

## Watermark Evaluation

### Baseline ROBIN

The baseline evaluation uses:

```bash
python inject_wm_baseline.py
```

### Predictor

The Predictor-based evaluation uses:

```bash
python inject_wm_with_predictor.py
```

Both pipelines retain the same underlying watermark generation, attack and verification logic so that the primary comparison concerns the method used to obtain the optimised prompt embedding.

---

## Unseen-Prompt Evaluation

Two unseen-prompt sets are included:

```text
new_unseen_prompts_20.txt
new_unseen_prompts_50.txt
```

Together, these contain **70 non-overlapping unseen prompts** used during the evaluation.

The final 50-prompt attack sweep can be executed on Windows PowerShell using:

```powershell
.\run_attack_sweep_50.ps1
```

The repository also contains:

```text
run_attack_sweep_20_and_100.ps1
```

which was used during the earlier experimental evaluation.

The attack sweeps compare baseline ROBIN and the Predictor under progressively stronger image manipulations.

Generated experimental results and logs are intentionally excluded from the repository.

---

## Predictor Validation

Additional experiments were performed to investigate whether the Predictor was learning a meaningful relationship between the original and optimised prompt embeddings.

A shuffled-target negative-control experiment deliberately mismatched the target optimised embeddings during training.

The correctly trained Predictor achieved substantially lower validation and test errors than the shuffled-target model.

This provides evidence that the Predictor relies on the relationship between the original CLIP embedding and the ROBIN optimised embedding rather than simply reproducing arbitrary target representations.

Utilities related to Predictor validation and prompt-overlap checking are included in the repository.

---

## Experimental Findings

The experiments indicate that replacing ROBIN's iterative prompt-embedding optimisation with the Residual Predictor can preserve comparable watermark robustness and perceptual image quality under the evaluated conditions.

Across the final 50-prompt attack sweep:

```text
Mean AUC
ROBIN:      0.9636
Predictor:  0.9648

Mean Accuracy
ROBIN:      0.9367
Predictor:  0.9383

Mean TPR @ 1% FPR
ROBIN:      0.7867
Predictor:  0.8083
```

Perceptual-quality metrics were also very similar:

```text
PSNR
ROBIN:      20.0568
Predictor:  20.0506

SSIM
ROBIN:      0.6129
Predictor:  0.6132

MS-SSIM
ROBIN:      0.7729
Predictor:  0.7734
```

These results should be interpreted as showing **comparable performance**, rather than evidence that the Predictor universally outperforms ROBIN.

---

## Robustness Boundary

The clearest degradation observed during the final evaluation occurred under strong Gaussian blurring.

At the strongest tested blur configuration:

```text
AUC
ROBIN:      0.7452
Predictor:  0.7480

TPR @ 1% FPR
ROBIN:      0.08
Predictor:  0.08
```

This indicates that severe blurring remains a significant robustness limitation of the underlying watermark verification pipeline.

---

## Web Application

The repository includes a Flask-based research prototype under:

```text
webapp/
```

The application provides:

- Predictor-based image generation
- ROBIN-based image generation
- Generated-image preview
- Image download
- Image copy functionality
- Direct verification of generated images
- Standalone image upload and verification
- ROBIN watermark detection
- Image Signature generation
- Image Signature verification
- Image-integrity checking
- Source information for signed images

The web application provides a browser-based demonstration of the research pipeline.

---

## Web Application Workflow

### Generation

```text
Text Prompt
     |
     v
Predictor / ROBIN
     |
     v
Stable Diffusion
     |
     v
Watermarked Image
     +
Image Signature
```

### Verification

```text
Uploaded Image
     |
     +-------------------------+
     |                         |
     v                         v
Watermark Check         Image Signature Check
     |                         |
     v                         v
ROBIN watermark          Signature validity
detection                + image integrity
```

Watermark detection and Image Signature verification are independent checks.

The watermark check determines whether the supported ROBIN watermark is detectable, while the Image Signature provides additional source and integrity information for images signed by the prototype.

---

## Running the Web Application

From the repository root:

```bash
cd webapp
```

Generate the local Image Signature keys if they do not already exist:

```bash
python generate_signature_keys.py
```

Then start the Flask application:

```bash
python app.py
```

The application normally runs locally at:

```text
http://127.0.0.1:5000
```

Stable Diffusion generation and ROBIN verification are computationally demanding, so GPU execution is strongly recommended.

---

## Image Signature Keys

Cryptographic key files are intentionally **not included in this repository**.

After cloning the repository, generate a local Image Signature key pair using:

```bash
cd webapp
python generate_signature_keys.py
```

The generated signing material is stored under:

```text
webapp/signature_keys/
```

This directory is excluded through `.gitignore`.

**Never commit the generated private signing key to a public repository.**

---

## Large Files and Generated Data

The following files and directories are intentionally excluded from version control:

- Python virtual environments
- Stable Diffusion model files
- ROBIN checkpoints
- Predictor checkpoints
- Training-pair tensors
- Generated clean images
- Generated watermarked images
- Uploaded images
- Experiment result directories
- Experiment logs
- Runtime prompt files
- Temporary images
- Image Signature cryptographic keys

These files must be generated or supplied locally when reproducing the complete system.

---

## Limitations

This project is a research prototype and has several important limitations.

- The Predictor approximates the result of ROBIN optimisation rather than reproducing the optimisation process exactly.
- The evaluation uses 70 unseen prompts across two evaluation sets and the attack configurations tested in this project.
- The results should not be interpreted as proof of robustness against every possible image manipulation.
- Severe Gaussian blurring substantially reduces watermark detection performance.
- ROBIN watermark verification remains dependent on the underlying ROBIN inversion and detection mechanism.
- The Image Signature mechanism provides source and integrity information only for images signed by this prototype.
- The Image Signature system is not a universal detector for arbitrary AI-generated images or arbitrary watermarking systems.
- Stable Diffusion generation and ROBIN verification remain computationally demanding and benefit substantially from GPU execution.
- The Flask application is a research demonstration rather than a production deployment.

---

## Foundation and Acknowledgement

This project is built on the original **ROBIN** research codebase:

**Huang, H., Wu, Y., & Wang, Q. (2024). ROBIN: Robust and Invisible Watermarks for Diffusion Models with Adversarial Optimization. Advances in Neural Information Processing Systems (NeurIPS 2024).**

Original ROBIN repository:

```text
https://github.com/Hannah1102/ROBIN
```

ROBIN paper:

```text
https://arxiv.org/abs/2411.03862
```

The original ROBIN implementation also acknowledges **Tree-Ring Watermarks** as an influence.

The Residual Prompt Embedding Predictor, extended attack evaluation pipeline, Flask research prototype, and Image Signature integration contained in this repository were developed as extensions to the ROBIN foundation for this dissertation project.

---

## Citation

If using the underlying ROBIN method or codebase, please cite the original ROBIN work:

```bibtex
@inproceedings{huangrobin,
  title={ROBIN: Robust and Invisible Watermarks for Diffusion Models with Adversarial Optimization},
  author={Huang, Huayang and Wu, Yu and Wang, Qian},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```

Please refer to the original ROBIN repository and paper for the authors' official implementation and documentation.

---

## Dissertation Context

This repository was developed as part of an MSc Artificial Intelligence dissertation investigating efficient and robust watermarking for generative AI images.

The central research question is whether ROBIN's computationally expensive iterative prompt-embedding optimisation can be approximated using a learned Residual Predictor while maintaining comparable watermark robustness and image quality under content-laundering attacks.

The experimental evaluation includes:

- **70 non-overlapping unseen prompts** across two evaluation sets;
- progressive attack-severity testing;
- baseline ROBIN vs Predictor comparisons;
- Predictor validation experiments;
- image-quality evaluation;
- watermark-verification evaluation; and
- a Flask-based research prototype demonstrating generation, watermark verification and Image Signature functionality.