# GAN on CIFAR-10: Airplane Image Generation

Deep Convolutional GAN implementation generating realistic airplane images from CIFAR-10 dataset.

## Quick Start

```bash
pip install tensorflow matplotlib numpy
python dcgan_cifar10.py
```

## Architecture

### Generator: Noise → 32x32 RGB Image
```
100D Noise → Dense(4×4×512) → Conv2DTranspose layers → 32×32×3 RGB
```

### Discriminator: Image → Real/Fake Classification  
```
32×32×3 RGB → Conv2D layers → Dense(1) → Binary output
```

## Key Results

- **Training**: 3,000 epochs on ~5,000 airplane images
- **Architecture**: Full convolutional design with batch normalization
- **Control Techniques**: Fixed noise consistency, smooth interpolation, magnitude scaling
- **Quality**: Progressive improvement in airplane feature learning

## Implementation Highlights

### DCGAN Improvements Over Basic GAN
- **Convolutional layers** replace dense layers for spatial feature learning
- **Batch normalization** stabilizes training dynamics  
- **LeakyReLU activations** prevent gradient problems
- **Dropout regularization** in discriminator

### Noise Vector Control

| Technique | Purpose | Result |
|-----------|---------|---------|
| Fixed Noise | Consistency testing | Identical outputs from same input |
| Interpolation | Smooth transitions | Morphing between airplane designs |
| Magnitude Scaling | Feature control | Adjustable image intensity/prominence |

### Training Configuration
- **Dataset**: CIFAR-10 airplane class (Class 0)
- **Batch Size**: 32 for training stability
- **Optimizer**: Adam (lr=0.0002, β₁=0.5)
- **Monitoring**: Loss tracking + sample generation every 200 epochs

## Technical Implementation

```python
# Generator architecture
def build_generator():
    # 100D noise → 4×4×512 → upsample to 32×32×3
    
# Discriminator architecture  
def build_discriminator():
    # 32×32×3 → downsample to 4×4×512 → binary classification
```



## Key Achievements

- **Modern Architecture**: Upgraded from dense-layer GAN to full DCGAN
- **Color Images**: Successfully generates 32×32 RGB airplane images  
- **Stable Training**: 3,000 epochs with consistent convergence
- **Latent Control**: Demonstrated multiple noise vector manipulation techniques

---

*Advanced generative modeling with convolutional neural networks and adversarial training.*
