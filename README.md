# Tensorflow Implementation of *LatentOut*
This repository provides a Tensorflow implementation of the *LatentOut* framework for anomaly detection. It is an extension of *VAEOut*, an unsupervised deep anomaly detection method exploiting Variational Autoencoders.

## Citation and Contact

- *LatentOut* (Machine Learning, 2022) [PDF](https://link.springer.com/article/10.1007/s10994-022-06153-4) and BibTex:

```
@article{angiulli2022mathrm,
  title={LatentOut: an unsupervised deep anomaly detection approach exploiting latent space distribution},
  author={Angiulli, Fabrizio and Fassetti, Fabio and Ferragina, Luca},
  journal={Machine Learning},
  pages={1--27},
  year={2022},
  publisher={Springer}
}
```

- *VAEOut* (Discovery Science, 2020) [PDF](https://link.springer.com/chapter/10.1007/978-3-030-61527-7_39) and BibTex:

```
@inproceedings{angiulli2020improving,
  title={Improving deep unsupervised anomaly detection by exploiting VAE latent space distribution},
  author={Angiulli, Fabrizio and Fassetti, Fabio and Ferragina, Luca},
  booktitle={International Conference on Discovery Science},
  pages={596--611},
  year={2020},
  organization={Springer}
}
```

If you would like to get in touch, you can write at luca.ferragina@unical.it
## Abstract
Anomaly detection methods exploiting autoencoders (AE) have shown good performances. Unfortunately, deep non-linear architectures are able to perform high dimensionality reduction while keeping reconstruction error low, thus worsening outlier detecting performances of AEs. To alleviate the above problem, recently some authors have proposed to exploit Variational autoencoders (VAE) and bidirectional Generative Adversarial Networks (GAN), which arise as a variant of standard AEs designed for generative purposes, both enforcing the organization of the latent space guaranteeing continuity. However, these architectures share with standard AEs the problem that they generalize so well that they can also well reconstruct anomalies. In this work we argue that the approach of selecting the worst reconstructed examples as anomalies is too simplistic if a continuous latent space autoencoder-based architecture is employed. We show that outliers tend to lie in the sparsest regions of the combined latent/error space and propose the VAEOut and LatentOut unsupervised anomaly detection algorithms, identifying outliers by performing density estimation in this augmented feature space. The proposed approach shows sensible improvements in terms of detection performances over the standard approach based on the reconstruction error.
