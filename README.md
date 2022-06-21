# Machine Leraning and Programming - Project 8
## Bogyeong Suh, Geon Kim, Sri Madhumitha Nelakanti, Tuli Barman

<img src="https://user-images.githubusercontent.com/73596975/174717850-938bf0c6-c254-432c-97f8-259a4099a2b0.png" width=430 height=353)>

### Description
In this repository, we provide the codes for project 8 'Control Logic Synthesis for Manufcaturing System Using Deep Reinforcement Learning'. We trained a deep reinforcement learning to create a scheduling of testing in manufacture process. We used DQN, parallel training, and customized reward function which resulted in the lower make-span compared to other rule-based algorithms.

The environment consists of 2 sections, A and B, each with 5 floors and 5 testers. An incoming pallet has to be tested on section A tester, and then, move on to section B tester, then exit. The testing time of all the testers follow a normal distribution. Pallets can choose which floor to be tested, and then move to different floors with lift. Testers, conveyor belts, and lift can only accommodate up to maximum 1 pallet at any instant. 

During the training, observation space containing the information of occupation of each place is saved as numpy arrays. Actions and rewards are given upon this observation space and the reward is fed into the DQN. 

The states, actions and rewards are set as below:

**States**

Status of all the testers and paths (occupied / idle)

**Actions**

Floor selection (once each, for section A and B)

**Rewards**

1) When action is given to choose all-occupied floor, _reward=-1_
2) When action is given to choose certain floor, crowdedness and crowd ranking are calculated using the number of pallets occupying testers or paths in each floor.
When RL chooses the least crowded floor, _reward=1_
When RL chooses the most crowded floor, _reward=-1_
In other cases, _reward=occupancy of testers in section A (or B), (range:0~1)_
3) When waiting, _reward=occupancy of testers in section A (or B), (range:0~1)_




### Code description
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

