# Multi-Agent path planning in Python

## Introduction

This repository consists of the implementation CBS of multi-agent path-planning algorithms in Python:

## Go to the virtual env
```
python3 -m venv .venv 
source .venv/bin/activate
```

## Dependencies

Install the necessary dependencies by running.

```shell
python3 -m pip install -r requirements.txt
```


### Conflict Based Search

Conclict-Based Search (CBS), is a multi-agent global path planner.

#### Execution

Run:

```shell
cd cbs
python3 cbs.py ../benchmark/inputs/input_3x3.yaml ../benchmark/outputs/output_3x3.yaml
```

#### Results

To visualize the generated results:

```shell
python3 visualize.py benchmark/inputs/input_3x3.yaml benchmark/outputs/output_3x3.yaml
```




#### References

- [Conflict-based search for optimal multi-agent pathfinding](https://www.sciencedirect.com/science/article/pii/S0004370214001386)
- [Suboptimal Variants of the Conflict-Based Search Algorithmfor the Multi-Agent Pathfinding Problem](https://ojs.aaai.org/index.php/SOCS/article/view/18315/18106)

