# FedLF

Official codes for ACML '24 research paper: FedLF: Adaptive Logit Adjustment and Feature Optimization in Federated Long-Tailed Learning.

### Dependencies

- python 3.7.9 (Anaconda)
- PyTorch 1.7.0
- torchvision 0.8.1
- CUDA 11.2
- cuDNN 8.0.4



### Dataset

- CIFAR-10
- CIFAR-100
- ImageNet-LT



### Parameters

The following arguments to the `./options.py` file control the important parameters of the experiment.

| Argument                    | Description                                       |
| --------------------------- | ------------------------------------------------- |
| `num_classes`               | Number of classes                                 |
| `num_clients`               | Number of all clients.                            |
| `num_online_clients`        | Number of participating local clients.            |
| `num_rounds`                | Number of communication rounds.                   |
| `num_epochs_local_training` | Number of local epochs.                           |
| `batch_size_local_training` | Batch size of local training.                     |
| `ipc`                       | Number of federated features per class.           |
| `lr_local_training`         | Learning rate of client updating.                 |
| `non_iid_alpha`             | Control the degree of heterogeneity.              |
| `imb_factor`                | Control the degree of imbalance.                  |



### Usage

Here is an example to run FedLF on CIFAR-10 with imb_factor=0.01:

```python
python main.py --num_classrs=10 \ 
--num_clients=20 \
--num_online_clients=8 \
--num_rounds=200 \
--num_epochs_local_training=10 \
--batch_size_local_training=32 \
--match_epoch=100 \
--ctr_epoch=300 \
--ipc=100 \
--lr_local_training=0.1 \
--lr_feature=0.1 \
--lr_net=0.01 \
--non-iid_alpha=0.5 \
--imb_factor=0.01 \ 
```

In Linux environments, here is an example to run CFedLF on CIFAR-10 with imb_factor=0.01 and save the output log to file:

```python
python main.py --num_classrs=10 \ 
--num_clients=20 \
--num_online_clients=8 \
--num_rounds=200 \
--num_epochs_local_training=10 \
--batch_size_local_training=32 \
--match_epoch=100 \
--ctr_epoch=300 \
--ipc=100 \
--lr_local_training=0.1 \
--lr_feature=0.1 \
--lr_net=0.01 \
--non-iid_alpha=0.5 \
--imb_factor=0.01 | tee creff_imb001_cifar10lt.log
```

