# ClinAlign

# 1.Getting Started
## Installation
1. Create  new conda environment:
 ```python
   conda create -n clinalign python=3.8
   conda activate clinalign
   ```
2. Install the dependencies in requirements.txt:
```python
pip install -r requirements.txt
```
# 2. Datasets Preparation
We use two publicly available radiology report generation datasets (MIMIC-CXR and IU X-Ray) in our paper.
- MIMIC-CXR V2.0.0: The corresponding X-Ray images can be downloaded from [MIMIC-CXR](https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/). The corresponding annotation files can be downloaded from the [PromptMRG](https://github.com/jhb86253817/PromptMRG) official repository. All MIMIC-CXR-related annotation and image files should be placed under the `data/mimic_cxr/` folder.
- IU X-RAY: The corresponding X-Ray images can be downloaded from [IU X-RAY](https://openi.nlm.nih.gov/faq) Official Website. The corresponding annotation files can be downloaded from the [PromptMRG](https://github.com/jhb86253817/PromptMRG) official repository. All ralated annotation and image files should be placed under the `data/iu_xray/` folder.
# 3. Download necessary files
- Bio_ClinicalBERT pre-trained weights: available on [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- Medical pre-trained ViT weights: available at [Google Drive](https://drive.google.com/file/d/1yMzChWuByT0Kxz3S5BYc8S2_5MVfi4Xn/view?usp=sharing)
- Visual Pattern Memory: available at [Google Drive](https://drive.google.com/file/d/1CRqxcEod1kFUjKFOQ2D6k4DprS6DQQec/view?usp=drive_link)
- Textual Pattern Memory: available at [Google Drive](https://drive.google.com/file/d/16Wlpm-91ssV0EvYx-ehH2N7cIOHM5BR5/view?usp=drive_link)

# 4. Training
Run `train_mimic_cxr.sh` to train a model on the MIMIC-CXR dataset.

Run `train_iu_xray.sh` to train a model on the IU_XRAY dataset.

Our all experiments are conducted on a single NVIDIA A40 GPU (48GB), and training takes approximately 24 hours.

# 5.Testing 
Run `test_mimic_cxr.sh` to train a model on the MIMIC-CXR dataset.

Run `test_iu_xray.sh` to train a model on the IU_XRAY dataset.

You can download our pretrained checkpoints for [Google Drive](https://drive.google.com/file/d/1o2UorBvpznmNn0oGiHGDo9B913iU6xxi/view?usp=sharing)

# 5.Acknowledge
We sincerely thank the contributions of the prior methods [R2Gen](https://github.com/zhjohnchan/R2Gen),[PromptMRG](https://github.com/jhb86253817/PromptMRG), [AM-MRG](https://github.com/Event-AHU/Medical_Image_Analysis/tree/main/AM_MRG).
