AsynFL
---
![AsynFL](/images/AsynFL.png)
* Synchronous schemes perform aggregation after receiving all local models (i.e., $N$). Asynchronous schemes perform aggregation as soon as it receives a model (i.e., 1) without waiting for other devices.
* We propose a semi-asynchronous aggregation scheme to achieve the trade-off between resource overhead and training performance. The server aggregates a certain number (i.e., $1\le M\le N$) of local models by their arrival order in each round.<br>
* Adjust the learning rate for each device according to its relative frequency of participating in aggregation.

Requirements
---
Pytorch v1.6 <br>
CUDA v10.0 <br>
cuDNN v7.5.0 <br>

Usage
---
We conduct experiments on a hardware prototype system, which consists of an AMAX deep learning workstation as the server and 30 NVIDIA Jetson TX2 devices.<br>
To reproduce experiments, please migrate `client_module` to embedded devices (e.g., NVIDIA Jetson TX2), and run the following command for the devices:<br>
```python
    python client.py --master_ip 'MASTER_IP'
```
Please replace an argument for `--master_ip` with your server IP and run the following command for the server:<br>
```python
    python server.py --epoch 500
```
