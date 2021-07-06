# Functional Magnetic Resonance Imaging data augmentation through conditional ICA

This repository includes the necessary code and models to reproduce the results of our recent paper:

**Functional Magnetic Resonance Imaging data augmentation through conditional ICA** <br>
*Badr Tajini, Hugo Richard, Bertrand Thirion* <br>
MICCAI 2021 <br>
Paper:  <br>

## Abstract 

Advances in computational cognitive neuroimaging research are related to the availability of large amounts of labeled brain imaging data, but such data are scarce and expensive to generate. While powerful data generation mechanisms, such as Generative Adversarial Networks (GANs), have been designed in the last decade for computer vision, such improvements have not yet carried over to brain imaging. A likely reason is that GANs training is ill-suited to the noisy, highdimensional and small-sample data available in functional neuroimaging.
In this paper, we introduce Conditional Independent Components Analysis (Conditional ICA): a fast functional Magnetic Resonance Imaging (fMRI) data augmentation technique, that leverages abundant restingstate data to create images by sampling from an ICA decomposition. We then propose a mechanism to condition the generator on classes observed with few samples. We first show that the generative mechanism is successful at synthesizing data indistinguishable from observations, and that it yields gains in classification accuracy in brain decoding problems. In particular it outperforms GANs while being much easier to optimize and interpret. Lastly, Conditional ICA enhances classification accuracy in eight datasets without further parameter tuning.

[WIP]
