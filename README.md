# CS5720 Neural Networks and Deep Learning — Home Assignment 3

**Student name:** Vinay Kumar Christina Rithika Nethakani
**Student ID:** 700778781
**Course / Term:** CS5720, Fall 2026, University of Central Missouri

## Repository contents
| File | Description |
|---|---|
| `part1_answers.md` | Part I answers (same content as below) |
| `part2_q1_convolution.py` | 2-D convolution implemented from scratch with NumPy |
| `q1_output.txt` | Output of the convolution program |
| `part2_q2_transfer_learning.py` | ResNet18 frozen feature extractor vs. fine-tuning (cat vs. dog, CIFAR-10) |
| `results_table.md`, `training_loss.png` | Generated when the transfer-learning script is run |

---

# Part I — Short-Answer Questions

## Question 1 — Convolution, Padding, and Stride
Input 32 × 32 × 3, 16 filters of size 5 × 5, S = 1, P = 0.

**a.** N = 32 (input height/width), F = 5 (filter size), S = 1, P = 0.

**b.** Nout = (N − F + 2P)/S + 1 = (32 − 5 + 0)/1 + 1 = **28**, so the feature map is 28 × 28.

**c.** Each of the 16 filters produces one 28 × 28 map, so the output volume is **28 × 28 × 16**.

**d.** To keep 32 × 32 with S = 1: (32 − 5 + 2P)/1 + 1 = 32, so 2P = 4 and **P = 2** ("same" padding, P = (F − 1)/2).

## Question 2 — CNN Architectures
**a. AlexNet:** It showed that a deep CNN trained on GPUs could win ImageNet (2012) by a large margin. It popularized ReLU activations, dropout, data augmentation, and GPU training.

**b. VGGNet:** Stacked 3 × 3 filters give the same receptive field as one large filter (two give 5 × 5, three give 7 × 7) with fewer parameters (27C² vs. 49C² for one 7 × 7 layer) and more non-linearities (extra ReLUs), making the network deeper and more discriminative.

**c. Inception module (GoogLeNet):** It applies 1 × 1, 3 × 3, and 5 × 5 convolutions and pooling in parallel and concatenates the outputs, capturing features at multiple scales. 1 × 1 convolutions reduce channels beforehand to keep computation low.

**d. ResNet:** Residual (skip) connections learn F(x) and output F(x) + x. This addresses the degradation/vanishing-gradient problem in very deep networks: gradients flow through the shortcuts and layers can easily learn an identity mapping, so deeper networks train better instead of worse.

## Question 3 — Representation Learning and Transfer Learning
**a. Representation learning:** Methods where a model automatically learns useful features from raw data instead of relying on hand-engineered features. CNNs learn them end-to-end.

**b. Lower vs. higher layers:** Lower layers have small receptive fields and see raw pixels, so they learn generic patterns such as edges, colors, and textures. Higher layers combine earlier features over larger receptive fields, learning complex, task-specific features (parts, shapes, whole objects).

**c. Transfer learning / fine-tuning:** Transfer learning reuses a model trained on a large source task (e.g., ImageNet) as the starting point for a new task. Fine-tuning further trains the pretrained weights (all or some layers) on the new data, typically with a small learning rate.

**d. Small dataset:** A deep CNN has millions of parameters, so training from scratch on little data overfits and needs heavy compute. A pretrained CNN already holds general features learned from millions of images, so it needs less data, trains faster, and generalizes better.

**e. Freezing vs. fine-tuning:** A frozen layer has fixed weights (`requires_grad = False`), so no gradients are computed and no updates happen. A fine-tuned layer remains trainable and is updated by backpropagation. Freezing is cheaper and safer with tiny datasets; fine-tuning is more flexible and often more accurate, but slower and more prone to overfitting.

---

# Part II — Programming Questions

## Question 1 — Convolution from Scratch
File: `part2_q1_convolution.py`. The program stores the input and 3 × 3 filter as NumPy arrays, slides the filter across the input (stride 1, padding 0), computes the element-wise multiply-and-sum (dot product) at each of the 9 valid positions, and prints the feature map and its shape. No built-in convolution function is used.

Run: `python part2_q1_convolution.py`

Output feature map:
```
[[4 3 4]
 [2 4 3]
 [2 3 4]]
```
Output shape: **(3, 3)**

**(f) Effect of stride 2:** The filter jumps two pixels at a time, so it visits fewer positions. Nout = (5 − 3)/2 + 1 = 2, so the output shrinks from 3 × 3 to **2 × 2** (`[[4 4],[2 4]]`). Larger strides downsample the feature map and reduce computation, but some spatial detail is skipped.

## Question 2 — Transfer Learning: Freeze vs. Fine-Tune
File: `part2_q2_transfer_learning.py`

**Setup:** Pretrained ResNet18 (ImageNet); dataset is CIFAR-10 restricted to two classes (cat vs. dog) with 1000 training and 250 test images per class, resized to 128 × 128; Adam optimizer (lr = 1e-3); **5 epochs for both experiments**.

- **Experiment A — Feature extraction:** load pretrained ResNet18, freeze all convolutional/base layers, replace the final `fc` layer with a 2-class layer, train only the new classifier.
- **Experiment B — Fine-tuning:** start from the pretrained network, unfreeze the last convolutional block (`layer4`), and train `layer4` plus the new classifier.

Run:
```
pip install torch torchvision matplotlib
python part2_q2_transfer_learning.py
```
The script prints the trainable parameter count, per-epoch loss, training time, and test accuracy for each experiment, and saves `training_loss.png`, `results_table.md`, and `results.json`.

### Training loss plot
![Training loss](training_loss.png)

### Comparison table
Trainable parameter counts are exact for ResNet18 with a 2-class head. Fill in the time and accuracy from your run.

| Method | Trainable Parameters | Training Time | Accuracy |
|---|---|---|---|
| Frozen Feature Extractor | 1,026 | 95.6s | 80.60% |
| Fine-Tuned Network | 8,394,754 | 125.1s | 86.00% |

### Discussion
The frozen feature extractor trains only 1,026 parameters (the new fc layer), while fine-tuning trains about 8.39 million (layer4 plus fc). Freezing lets PyTorch skip gradient computation and optimizer updates for the frozen layers, so each epoch is cheaper and training time is lower. The frozen model relies on generic ImageNet features, which work well when the new task is similar to ImageNet. Fine-tuning lets layer4 adapt its high-level features to the new classes, so it typically reaches lower training loss and higher accuracy. The cost is longer training and a higher risk of overfitting on a small dataset. _<Add 1–2 sentences with your actual measured times and accuracies, and whether fine-tuning's gain justified its extra cost.>_
