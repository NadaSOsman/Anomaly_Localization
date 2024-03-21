# Temporal Divide-and-Conquer Anomaly Actions Localization

## Requirments
1. Get the features available in [SlowFast Features](https://drive.google.com/file/d/1N6M3SgLrNCsRE8fUaa3Qn8Mtkp40jRaI/view?usp=sharing). 
2. Setup the environment:
   ``` 
   conda env create -f env.yml
   conda activate AD
   ```
   
## Training
1. All training configurations can be found in `configs.yaml`.
2. To train the model, run the following:
    
    `python3 run.py --configs_file configs.yaml`
  
3. To resume training:
    
   `python3 run.py --configs_file configs.yaml --resume
  
## Evaluation
1. To evaluate the per-video performance of the model, run the following:
    
    `python3 run.py --configs_file configs.yaml --test`
   
2. For anomaly localization evaluation, run:
   
    `python3 anomaly_localization.py --configs_file configs.yaml`

## Pre-trained Model
You can find [here](https://drive.google.com/file/d/1BlkpLjabSeDjZnjyjyDYBRpepW_zXIcR/view?usp=sharing) our pre-trained checkpoint.
