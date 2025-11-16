# Multi-Agent path planning in Python

## Introduction

This repository consists of the implementation CBS of multi-agent path-planning algorithms in Python:

## Go to the virtual env
```
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

``` 
cd ./centralized/cbs
python3 cbs.py input.yaml output.yaml
```

#### Results

To visualize the generated results:

``` shell
python3 ../visualize.py input.yaml output.yaml
```




#### Reference

- [Conflict-based search for optimal multi-agent pathfinding](https://www.sciencedirect.com/science/article/pii/S0004370214001386)

