# Facial Expression Recognition with RWKV

Facial expression recognition is an important task in computer vision with various real-world applications, such as human-computer interaction. This project explores the possibilities of utilizing the Receptance Weighted Key Value (RWKV) architecture in facial expression recognition. The objective is to integrate RWKV into the current state-of-the-art (SOTA) architecture, PosterV2.

## Project Steps
1. 📚 **Literature Review:** Understand the current research on facial expression recognition, focusing on RWKV and PosterV2. ✅
2. ⚙️ **Code Setup:** Get the code running for RWKV and PosterV2. (🔄 In Progress)
3. 🏗️ **Architecture Integration:** Replace the transformer in PosterV2 with the RWKV layer. This is the crucial part of the project, requiring significant effort due to its complexity. 
4. 🏋️‍♂️ **Training:** Train the new architecture. 
5. 📊 **Evaluation:** Compare the results achieved by the integrated RWKV architecture. 

## PosterV2 Architecture
PosterV2 is an enhanced version of the state-of-the-art PosterV1 architecture for facial expression recognition. It achieves SOTA performance with minimal computational costs by combining facial landmark and image features using a two-stream pyramid cross-fusion design. Improvements in PosterV2 include a window-based cross-attention mechanism, removal of the image-to-landmark branch, and multi-scale feature extraction. Extensive experiments on standard datasets demonstrate that PosterV2 achieves SOTA Facial Expression Recognition (FER) performance with efficient computational requirements.

## RWKV Architecture
RWKV is a novel model architecture that combines the efficient parallelizable training of Transformers with the efficient inference of recurrent neural networks (RNNs). It employs a linear attention mechanism and allows the model to be formulated as either a Transformer or an RNN. RWKV is the first non-transformer architecture to scale to tens of billions of parameters, performing on par with similarly sized Transformers. This suggests potential for leveraging RWKV to create more efficient models in future work.

## Usage
Instructions for using the project will be provided soon.

## Citation
Citation details will be provided soon.