# A Dual Augmentation Framework for Domain Generalization with both Covariate and Conditional Distribution Shifts

Deep learning models often suffer performance degradation due to domain shifts between training and testing data distributions.
Domain Generalization (DG) addresses this challenge by leveraging knowledge from multiple source domains to enhance model generalization capabilities for unseen domains.
Data augmentation is a primary method in DG, aiming to improve model generalizability by augmenting source domain data.
However, existing methods mainly focus on covariate shifts, neglecting conditional distribution shifts and thus limiting augmented data diversity.
To address this limitation, we propose the Dual Augmentation Framework (DAF), incorporating two sub-augmentation frameworks: Covariate Augmentation (CovAug) and Conditional Augmentation (ConAug).
CovAug is adaptable to existing methods and enriches source data while maintaining similar variability.
For ConAug, we develop an Adversarial Class Transformation Network (ACTNet), which augments source data by introducing conditional distribution shifts through adversarial training with relative distance loss, exploring new regions of the feature space while preserving semantic consistency.
Furthermore, a diversity-based augmentation strategy that adjusts the perturbation weight of augmented data based on learnt embedding diversity is proposed to further improve the effectiveness of DAF.
Ablation studies confirm the effectiveness of each sub-framework, while visualizations provide deeper insights into DAF.

# Baseline
* ERM
* 2018 - CrossGrad - [Generalizing Across Domains via Cross-Gradient Training](https://openreview.net/forum?id=r1Dx7fbCW)
* 2020 - DDAIG - [Deep Domain-Adversarial Image Generation for Domain Generalisation](https://arxiv.org/abs/2003.06054)
* 2021 - MixStyle - [Domain Generalization with MixStyle](https://openreview.net/forum?id=6xHJ37MVxxp)
* 2022 - DomainMix - [Dynamic Domain Generalization](https://arxiv.org/abs/2205.13913)
* 2022 - EFDMix - [Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization](https://arxiv.org/abs/2203.07740)
* 2023 - CoOp - [Learning to Prompt for Vision-Language Models](https://openreview.net/forum?id=OgCcfc1m0TO)
* 2023 - RISE - [A Sentence Speaks a Thousand Images: Domain Generalization through Distilling CLIP with Language Guidance](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_A_Sentence_Speaks_a_Thousand_Images_Domain_Generalization_through_Distilling_ICCV_2023_paper.html)
* 2024 - SSPL -  [Symmetric Self-Paced Learning for Domain Generalization](https://ojs.aaai.org/index.php/AAAI/article/view/29639)

# Datasets
* Digits
* PACS
* OfficeHome
* VLCS
* Terra Incognita
* NICO++
* DomainNet

# Sample Command

python train.py

                --gpu 1                                                 # Specify device
                --seed 995                                              # Random Seed
                --output-dir output/DAF-RN50-NICO-autumn                # Output directory 
                --dataset NICO                                          # Specify dataset
                --source-domains dim grass outdoor rock water           # Source Domains
                --target-domains autumn                                 # Target Domain
                --model DAF                                             # Model for training
                --model-config-file config/daf.yaml                     # Config file for model

