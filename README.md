# Machine Leraning and Programming - Project 8
## Bogyeong Suh, Geon Kim, Sri Madhumitha Nelakanti, Tuli Barman

### Description
In this repository, we provide the codes for project 8 'Control Logic Synthesis for Manufcaturing System Using Deep Reinforcement Learning'. 
The following codes have two steps:
1. Training the RL model
2. Checking the results by rendering the simulation

**Python 3.7** is used, and please refer to **requirements.txt** for installing the packages needed for running the code. 

- **envs** \
-The main components and environment that structure our project are saved in this folder. \
-Pallets, Testers, Lifts, and all the other components are defined in the codes in envs folder. \
-States and Actions are defined in **pallet.py**, and Reward is defined in **floors_env.py**.

- **stable_baselines** \
-The source code of Deep-Q Networks utilzied as RL model is saved in stable_baselines/deepq/dqn.py\
-We added the codes for parallel training of two models.

- **utils** \
-Codes for saving the best model during training, and setting of hyperparameters are saved here. 

- **etc** \
-**test.py**, **visualize.py** were used for debugging the code and doing some experiments with the code.
These codes are not required for the actual training and testing the RL model.

---
### 1. Training
After structuring the environments with **requirements.txt**, run the **main.py**
```
python main.py --prefix Test1 
```

### 2. Check the result
When the best model is saved in Test1, run the **play.py** and select the Test1 folder. 
You can click on the folder after you run the below command. 
```
python play.py
```
This code will simulate using the best model saved in Test1, and save a mp4 video showing the simulation. 
Additionally, you can check the total required timesteps, which are noted as 'ELAPSED SIM-TIME' and printed after the simulation is done.
The video will be saved with the name of 'Test1.mp4'

