# Channel Tokenizer Integration

This document explains how the channel-tokenizer idea was integrated into the image transmission pipeline, what problem it solves, and how the end-to-end system is organized.

The goal is robustness: a single receiver should handle a family of channel conditions without needing a separate full retrain for every channel type.

## 1. What The Original Pipeline Does

The project starts from a clean image and does not transmit that image directly through the channel. Instead, it uses a reduced-resolution channel-facing representation first.

The original flow is:

1. Start with a clean full-resolution image.
2. Downsample it to a small spatial grid.
3. Pass that small image through the channel.
4. Add the channel corruption for the chosen channel family and SNR.
5. Upsample the corrupted small image back to the encoder input size.
6. Feed the noisy upsampled image into the Swin encoder.
7. Convert the resulting latent into a clean image with the decoder, optionally after diffusion refinement.

The important point is that the small image is the actual channel-facing signal. It is the object that is physically distorted by fading, noise, and other impairments. The encoder and decoder then work on a learned latent representation of that corrupted observation.

## 2. Why The Small Noisy Image Matters

The small noisy image is not an implementation detail. It is the channel observation.

That is useful for two reasons:

1. The channel distortion is visible in image space before the encoder compresses anything away.
2. The receiver can learn to infer channel family and channel strength from the corruption pattern itself.

In other words, the model is not required to guess the channel from a hidden variable. It receives the noisy image that the channel actually produced, and the receiver-side adaptation module learns from that evidence.

## 3. Where The Channel Tokenizer Fits

The channel tokenizer is inserted on the receiver side.

Conceptually, the receiver now has two jobs:

1. Recover a usable latent or intermediate representation from the distorted observation.
2. Summarize the channel state in a compact token that can guide routing or refinement.

The tokenizer does not replace the Swin encoder, and it does not replace the diffusion model. It sits beside them and helps the receiver adapt to the current channel condition.

The overall receiver-side flow is:

1. The channel produces a corrupted observation.
2. The receiver extracts a compact channel signature from that observation.
3. That signature is quantized into a discrete token.
4. A router uses the token to select or mix expert blocks.
5. The selected receiver path restores a latent suitable for decoding.
6. If diffusion is enabled, the token also conditions the latent refinement model.

## 4. Channel Tokenizer Design
## 4.1 Compared To SemCast

The tokenizer follows the same broad idea as SemCast, but it is not identical. The paper’s tokenizer is a receiver-side channel representation learned from the received signal. In this project, the tokenizer is adapted to the image pipeline and reads the receiver-side corrupted observation that comes from the image transmission path.

That difference matters:

1. SemCast is closer to a physical-layer channel tokenizer.
2. This project is a higher-level tokenizer for image-coded observations.
3. The token is used not only to describe the channel, but also to guide latent restoration and diffusion refinement.

So the shared principle is channel-aware routing from a discrete signature, but the object being tokenized is different.


The tokenizer is a SemCast-style idea adapted to this image pipeline.

The module is built from four pieces:

1. A lightweight encoder that reads the receiver-side corrupted representation.
2. A vector quantizer that turns the continuous channel signature into a discrete token.
3. A router that maps the token to expert weights.
4. A small expert bank that specializes in different channel regimes.

The purpose of the discrete token is to avoid the receiver having to memorize a separate network for every possible channel realization. The token acts like a compact, learned channel descriptor.

### 4.1 Token extraction

The tokenizer reads the receiver-side distorted representation and compresses it into a lower-dimensional signature.

This is not a pilot-based estimator. There are no explicit pilots that hand the model the channel state. The tokenizer learns to infer the channel from the pattern of corruption that remains in the received signal.

### 4.2 Vector quantization

The continuous signature is passed through a codebook.

This has two benefits:

1. It forces the model to represent channel conditions with a finite set of prototypes.
2. It makes the token stable and easy to route on.

This is important for mixed-channel training, because the same channel family can appear with different SNRs and minor impairments. A discrete codebook encourages consistent channel grouping.

### 4.3 Router and experts

The router turns the token into a distribution over experts.

Each expert is a lightweight residual adapter. The final restored signal is a weighted mixture of the expert outputs. This gives the model a controlled way to specialize without duplicating the full backbone.

The router is also regularized so that it does not collapse onto a single expert for every input.

## 5. How This Connects To The Image Pipeline

The tokenizer only makes sense because the receiver sees a corrupted image or a corrupted latent that came from the image pipeline.

The pipeline is:

1. Clean image.
2. Small channel-facing image.
3. Channel corruption.
4. Receiver observation.
5. Channel tokenization.
6. Latent restoration.
7. Optional diffusion refinement.
8. Final image reconstruction.

The original small-image transmission path provides the channel evidence. The tokenizer uses that evidence to decide how the receiver should behave.

That is the conceptual bridge between the original project and the new channel-adaptive receiver.

## 6. Training Strategy

The system is trained in stages.

### 6.1 Transport stage

The transport stage trains the transmitter and the receiver together.

The receiver learns:

1. How to tokenize the channel observation.
2. How to route the token into the right expert mixture.
3. How to reconstruct a usable latent from the corrupted input.

The losses used here encourage:

1. Latent reconstruction.
2. Codebook usage.
3. Balanced expert routing.
4. Channel-discriminative token formation.

### 6.2 Diffusion stage

The diffusion stage freezes the channel front end and trains the latent denoiser.

The denoiser gets:

1. A noisy latent.
2. The receiver-restored latent as a condition.
3. The channel token as a conditioning signal.

The model learns to refine the latent toward the clean target.

### 6.3 Joint stage

The joint stage lets the whole path adapt together.

This is the best setting for the strongest overall performance because:

1. The receiver adapts to the needs of the diffusion model.
2. The diffusion model learns the actual distribution of receiver outputs.
3. The tokenizer remains useful across different channel families.

## 7. Why This Is Better Than A Single Fixed Receiver

A fixed receiver assumes one channel behavior or tries to absorb all channel variation into one undifferentiated representation.

The tokenizer-based approach does better because it introduces an explicit intermediate representation of channel state.

That gives the model:

1. Better channel awareness.
2. Better specialization without full retraining.
3. Cleaner ablation stories.
4. A way to support mixed channels in a single system.

## 8. What Was Changed In Practice

The integration did not require changing the front-end image pipeline into something completely different.

Instead, the receiver-side adaptation path was upgraded so that channel corruption can be summarized and exploited explicitly.

The major practical changes were:

1. A tokenizer that reads the received observation.
2. A discrete codebook for channel signatures.
3. A router for expert selection.
4. Lightweight expert blocks for specialization.
5. Conditioning of the diffusion refinement model with the token.
6. Mixed-channel training under Sionna-based channel simulation.

## 9. Evaluation Logic

The system is evaluated with three main questions in mind:

1. How good is the transport-only reconstruction?
2. Does the tokenizer improve diffusion-based refinement?
3. Does the system remain stable across different channel families?

The reported metrics are:

1. PSNR.
2. SSIM.
3. Latent cosine similarity.
4. Latent CKA.

The ablations are meant to isolate the effect of:

1. The tokenizer.
2. The discrete codebook.
3. Expert routing.
4. The diffusion refinement stage.

## 10. What Readers Should Take Away

The integration is conceptually simple even if the implementation has several parts.

The original pipeline already had a channel-facing small noisy image and a latent reconstruction path. The new contribution is to make the receiver aware of the channel condition explicitly by extracting a discrete token from the received observation and using that token to guide specialization and diffusion refinement.

The practical result is a receiver that is no longer forced to treat every channel the same way.
