# Reinforcement learning with LoRA for targeted antimicrobial peptide design
## Abstract
This section will be completed after publication.

## Model weight and data
[Google drive link](https://drive.google.com/drive/folders/1XlMEKanxb5Tlr0YyN3eEiOPdHtmteCgz?usp=drive_link)


**AMP classifier**  
This folder into the `AMP-RL` folder


**MIC predictor**  
This folder into the `AMP-RL` folder


**Pretrained model ckpt**   
This folder into the `ckpt/Pretrain` folder


**PeptideAtlas data**  
This csv file into the `data` folder


## Model Implementation
The model implemented for the pretraining & reinforcement learning in `GPT_PeptideAtlas_Pretraining.ipynb` and `AMP_RL Reinforcement learning.ipynb` respectively
```
# environment setting
$ git clone https://github.com/GIST-CSBL/AMP-RL.git

$ cd AMP-RL

$ conda create -n AMP-RL python=3.9

$ conda activate AMP-RL

$ pip install -r requirements.txt

$ pip install ipykernel

$ python -m ipykernel install --user --name AMP-RL --display_name "AMP-RL"
```

## License
The source code in this repository is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/). See the `LICENSE` file for more information.

## Contact
Juntae Park (nuwana9876@gm.gist.ac.kr)


Daehun Bae (qoeogns09@gm.gist.ac.kr)


Hojung Nam* (hjnam@gist.ac.kr)


*Corresponding Author
