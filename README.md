# MARL-based policy optimization of wind farm control task

This repo is the code implementation of the paper titled "Reinforcement Learning-based Control of Wind Farm Composed of Hydrostatic Wind Turbines". In this paper, a novel MARL method is proposed to control the hydrostatic wind turbines (HWT) in a wind farm to maximize its power generation. The structure of this repo is:

## Dependency

1. Python3  (include numpy, tensorflow1.0 etc.)
2. FASTFarm (include OpenFAST)

## Simulator

[FASTFarm](https://github.com/OpenFAST/openfast): simulte the dynamics of a wind farm

However, FASTFarm uses the gearbox-based wind turbines and we replace it with the hydrostatic wind turbines. Please see the file `./Gearbox2hydrostatic_transmission.md` for detailed modifications.

There are two wind farm cases used in the simulation:
1. Three HWT in a wind farm (`./fast-farm/Three_Turbines`)
2. Six HWT in a wind farm (`./fast-farm/Six_Turbines`)

The simulation can be swiched by changing Line 90-103 in `./train.py`

## Data process

`./WriteData.py` file can write the weight of the policy network to the input file of OpenFAST.

`./ReadData.py` file can transform the output of OpenFAST into the samples used for training the MARL algorithm.

## Multi-agent policy optimization

`mapo.py` file contains the proposed MARL algorithm named MAPO.

`train.py` file contains the method to train MAPO.

The method to run the code:

```
mkdir policy learning_curves
python3 train.py --save-dir "./policy/" --plots-dir "./learning_curves"
```

Please use the following command to see other input parameters of the `train.py` file.

```
python3 train.py --help
```

## Plot

`plot.py` shows the method how to plot figures in the paper according to the outputs of `train.py` file.
`figure` directory includes the figures in the paper

