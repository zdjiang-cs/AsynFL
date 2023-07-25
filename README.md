AsynFL
---
![AsynFL](/images/AsynFL.png)
* Synchronous schemes perform aggregation after receiving all local models (i.e., $N$). Asynchronous schemes perform aggregation as soon as it receives a model (i.e., 1) without waiting for other clients.
* We propose a semi-asynchronous aggregation scheme to achieve the trade-off between resource overhead and training performance. The server aggregates a certain number (i.e., $1\le M\le N$) of local models by their arrival order in each round.<br>
* Adjust the learning rate for each client according to its relative frequency of participating in aggregation.

Requirements
---
Pytorch v1.6 <br>
CUDA v10.0 <br>
cuDNN v7.5.0 <br>

Usage
---
Initialize the parameters in `start_train.py`, including model types, dataset types, learning rate, batch size, etc.<br>
Run an experiment with the following command:<br>
```python
    python start_train.py
```
