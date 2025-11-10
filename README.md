# eBRoS
This package is used to roster a given time-bound tasks to energy constraint resources to generate an optimal rostering systems that minimizes the total operational costs and total inactivity (idle) time.

# Installation
To install the dependencies required for this version run the below command from the root directory of this project.

`pip install -r requirements.txt`
## To run the MIP Results from the simulated study using CPLEX OPL IBM Optimizer 
- Open the directory

  ```cd CPLEX-MIP && python cplex_visualizer.py```

# To run the Constructive Heuristic Search-SA eRoSB (Metaheuristic)
- Open the folder and run the following command:
  
  ```cd CHS-SA && python CHS-SA.py```
