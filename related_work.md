# Related Work

## SVG Generation

**LLM4SVG** (Xing et al., CVPR 2025) introduces 55 learnable semantic tokens that replace raw SVG tags and attributes, preventing subword fragmentation of structural markup. These tokens are integrated into a modular LLM architecture combining geometric, appearance, and language streams. The SVGX dataset (250K+ SVGs, 580K instructions) and the semantic token vocabulary used in our project originate from this work.

**Chat2SVG** (Wu et al., CVPR 2025) is a hybrid framework that uses an LLM to produce SVG templates from basic geometric primitives, then refines them via dual-stage optimization guided by an image diffusion model. Unlike our approach, refinement operates in pixel space through score distillation rather than directly predicting coordinates.

**SVGFusion** (Xing et al., 2024) learns a continuous latent space from both SVG code and rasterized images via a Vector-Pixel Fusion VAE, then generates latent codes with a Diffusion Transformer conditioned on text. This is the closest prior work to diffusion-based SVG coordinate generation, though it operates in a learned latent space rather than directly on SVG coordinates.

**VectorFusion** (Jain et al., CVPR 2023) pioneered using pretrained pixel-space diffusion models for SVG generation via Score Distillation Sampling (SDS), optimizing Bezier curve parameters through a differentiable rasterizer. **SVGDreamer** (Xing et al., CVPR 2024) and **DiffSketcher** (Xing et al., NeurIPS 2023) extend this SDS paradigm with improved optimization, editing support, and sketch synthesis at varying abstraction levels.

**DeepSVG** (Carlier et al., NeurIPS 2020) is a hierarchical VAE that disentangles high-level shapes from low-level SVG path commands, predicting shapes non-autoregressively. One of the earliest deep learning approaches to directly model SVG path coordinates.

**IconShop** (Wu et al., SIGGRAPH Asia 2023) serializes SVG paths and text descriptions into a single token sequence and trains an autoregressive transformer for text-conditioned icon generation. Key prior work on treating SVG generation as sequence prediction with tokenized representations.

**StarVector** (Rodriguez et al., 2024) is a vision-language model treating SVG generation as code generation, integrating a ViT encoder with an LLM. Introduces SVG-Stack (2M+ SVGs) and SVG-Bench, the current standard evaluation benchmark for SVG understanding and generation.

**Im2Vec** (Reddy et al., CVPR 2021) learns to produce vector graphics from raster supervision alone, encoding shapes as deformations of topological disks through a differentiable rasterization pipeline.


## Flow Matching and Continuous Diffusion

**Flow Matching** (Lipman et al., ICLR 2023) introduces a simulation-free framework for training Continuous Normalizing Flows (CNFs) by regressing vector fields along fixed conditional probability paths. This is the foundational algorithm our project builds upon, using linear interpolation paths between noise and data.

**Rectified Flow** (Liu et al., ICLR 2023) learns ODEs that follow straight paths connecting two distributions, with an iterative "reflow" procedure that straightens trajectories for efficient few-step generation. Provides the theoretical grounding for linear flow matching paths.

**Conditional Flow Matching** (Tong et al., ICLR 2024) generalizes CFM with a simulation-free training objective that does not require Gaussian source distributions. The OT-CFM variant creates simpler, more stable flows and is the practical training framework underlying most flow matching implementations.

**Discrete Flow Matching** (Gat et al., NeurIPS 2024) extends flow matching to discrete data via probability paths that interpolate between source and target distributions with learned posteriors. Scales to 1.7B parameters for code generation. Relevant as the discrete counterpart to our continuous coordinate flow matching.

**FlowLLM** (Joshi et al., NeurIPS 2024) fine-tunes an LLM to learn a base distribution of meta-stable crystals, then uses Riemannian flow matching to iteratively refine atomic coordinates and lattice parameters. **Most directly related to our architecture**: it combines a pretrained language model with flow matching for structured coordinate prediction.

**DiffuLLaMA** (Gong et al., ICLR 2025) converts pretrained autoregressive models (GPT-2, LLaMA) into diffusion models via continual pretraining, demonstrating that AR model weights transfer effectively to diffusion objectives. Precedent for adapting pretrained language models to diffusion-based generation.

**FrameFlow** (Yim et al., ICLR 2024) and **FoldFlow** (Bose et al., NeurIPS 2023 Workshop) apply flow matching on SE(3) manifolds for protein backbone generation, achieving faster and higher-quality structure generation than diffusion baselines. Demonstrates flow matching for structured coordinate prediction in scientific domains.

**FlowMol** (Dunn et al., ICML 2024) jointly applies flow matching over continuous 3D coordinates and categorical atom types for molecule generation, defining conditional vector fields for each modality. Analogous to our setting of continuous coordinates conditioned on discrete SVG tokens.


## Diffusion for Language and Discrete Data

**Diffusion-LM** (Li et al., NeurIPS 2022) pioneered continuous diffusion in word embedding space, iteratively denoising Gaussian vectors into word vectors. Enables gradient-based controllable text generation for fine-grained syntax and semantic constraints.

**CDCD** (Dieleman et al., 2022) applies continuous-time, continuous-space diffusion to categorical data by adding Gaussian noise to token embeddings, with score functions obtained through vocabulary embedding interpolation. Closely related to our approach of applying continuous flow matching to discrete token positions.

**D3PM** (Austin et al., NeurIPS 2021) generalizes diffusion to discrete state spaces with structured transition matrices, connecting diffusion to autoregressive and masked language modeling. **MDLM** (Sahoo et al., NeurIPS 2024) shows that masked discrete diffusion with modern training practices approaches autoregressive perplexity.

**SSD-LM** (Han et al., ACL 2023) generates text semi-autoregressively via diffusion on the vocabulary simplex, enabling classifier guidance. **Difformer** (Gao et al., NAACL 2024) applies denoising diffusion Transformers in continuous embedding space with anchor loss for text generation.


## Backbone and Adaptation

**MarkupLM** (Li et al., ACL 2022) is a BERT-style model pretrained on HTML/XML, encoding text content jointly with markup structure via XPath-based positional embeddings. We use it as a pretrained backbone because SVG is itself a markup language and the model already understands tag/attribute structure.

**LoRA** (Hu et al., ICLR 2022) injects trainable low-rank decomposition matrices into frozen Transformer layers, reducing trainable parameters by orders of magnitude while matching full fine-tuning performance. We apply LoRA to the attention layers of MarkupLM, keeping the pretrained representations intact while adapting the model for flow-based coordinate prediction.
