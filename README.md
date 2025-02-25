# GL-POMO
For "Local Topological Information as a Powerful Enhancer for Generalizable Neural Method in Travelling Salesman Problem"

Accepted by AAMAS2025.

## Dependencies
GPU: NVIDIA RTX 3090
```bash
python=3.9.19
torch=1.12.1
cudatoolkit=11.6.0
numpy=1.26.4
pytz=2024.1
matplotlib=3.9.0
```

## Datasets
The training and test datasets can be downloaded from Google Drive:
```
https://drive.google.com/drive/folders/1fpu4puWOHVB5rsa5uihhu1tM95YTm7U3?usp=sharing
```
- Download the datasets and put them to the path **./GL-POMO/TSP/data**
## Evaluation 
Results of Table1 for Our models 

```bash
cd ./GL-POMO/TSP
# perform evaluation
python validate_ours.py
```

## Training
```bash
cd ./GL-POMO/TSP
# You can edit the setting on the train_ours.py 
# perform training
python train_ours.py
```

# Acknowledgements
We would like to thank the following repositories, which are baselines of our code:
- https://github.com/wouterkool/attention-learn-to-route
- https://github.com/yd-kwon/POMO
- https://github.com/alstn12088/Sym-NCO

We employ the dataset from https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD to train TSP on uniform condition.
And we use the instance generator from https://github.com/jakobbossek/tspgen and https://github.com/Kangfei/TSPSelector
