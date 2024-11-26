# A implementation for the paper "Dual-Modality Visual Feature Flow for Medical Report Generation"

## Requirements

- `Python >= 3.6`
- `Pytorch >= 1.7`
- `torchvison`
- [Microsoft COCO Caption Evaluation Tools](https://github.com/tylin/coco-caption)
- [CheXpert](https://github.com/stanfordmlgroup/chexpert-labeler)

## Data

### Gird features

Download IU and MIMIC-CXR datasets, and place them in `data` folder.
- IU dataset from [here](https://iuhealth.org/find-medical-services/x-rays)
- MIMIC-CXR dataset from [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

### Region features

The region features is from [https://github.com/MILVLG/bottom-up-attention.pytorch#Training](https://github.com/MILVLG/bottom-up-attention.pytorch#Training)

## Training and Testing

- The validation and testing will run after training.
- The model will be trained using command：
    - $dataset_name:
        - iu: IU dataset
        - mimic: MIMIC dataset

          
     ```
     bash train_{$dataset_name}.sh
     ```

### Contact
* If you have any questions, please post them on the issues page or email at tangquan6719@stu.cwnu.edu.cn
