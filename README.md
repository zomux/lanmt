LaNMT: Latent-variable Non-autoregressive Neural Machine Translation with Deterministic Inference
===



```diff
- WARNING: Still doing code refactoring, please wait until this message disappears :)
```


LaNMT implements a latent-variable framework for non-autoregressive neural machine translation. As you can guess from the code, it's has a simple architecture but powerful performance. For the details of this model, you can check our paper on Arxiv https://arxiv.org/abs/1908.07181 . To cite the paper:

```
@article{Shu2019LaNMT,
  title={Latent-Variable Non-Autoregressive Neural Machine Translation with Deterministic Inference using a Delta Posterior},
  author={Raphael Shu and Jason Lee and Hideki Nakayama and Kyunghyun Cho},
  journal={ArXiv},
  year={2019},
  volume={abs/1908.07181}
}
```
In this model, we learn a set of continuous latent variables ![z](https://latex.codecogs.com/png.latex?z) to capture the information and intra-word dependencies of the target tokens. Intuitively, if the model is perfectly trained and the target sequence can be fully reconstructed from the latent variables without error, then the translation problem becomes a problem of finding adequate ![z](https://latex.codecogs.com/png.latex?z). This is illustrated in the picture below, which shows the relations among ![x](https://latex.codecogs.com/png.latex?x), ![z](https://latex.codecogs.com/png.latex?z)and ![y](https://latex.codecogs.com/png.latex?y).

<p align="center">
<img src="https://i.imgur.com/qh7sPlB.png" width="400px"/>
</p>

In paractice, we force the latent variables to have very low dimensions such as 8. Obviously, handling things in a low-dimension countinuous space is easier comparing to a high-dimension discrete space.

Our model is trained by maximizing the following objective, which is a lower bound of log-likehood. We  call it *evidence lower bound* (ELBO). The first part is a reconstruction loss that makes sure you can predict target sequence from ![z](https://latex.codecogs.com/png.latex?z). The second part is a KL divergence, which makes the ![z](https://latex.codecogs.com/png.latex?z) more predictable given the source sequence.

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\log%20p(y|x)%20\geq%20\mathbb{E}_{z%20\sim%20q(z|x,y)}%20\Big[\log%20p(y|x,z,l_y)%20+%20\log%20p(l_y|z)\Big]%20-%20\mathrm{KL}\Big(q(z|x,y)||p(z|x)\Big)" />
</p>
 
Now for the parameterization, the model is implemented with the architecture in the picture below. Does it appear to be more complicated comparing to a standard Transformer? Well, you are now computing four probabilities instead of only ![p(y|x)](https://latex.codecogs.com/png.latex?p(y|x)). However, as the model is basically reusing the Transformer modules such as self-attention and cross-attention, so it's still pretty easy to implement. 

<p align="center">
<img src="https://i.imgur.com/a3x9tni.png" width="400px"/>
</p>

One thing special about this model is that the number of latent variables is always identical to the source tokens, as you can guess from the first figure in this post. As each ![z_i](https://latex.codecogs.com/png.latex?z_i) is a continuous vector, ![z](https://latex.codecogs.com/png.latex?z)is a ![L_x by D](https://latex.codecogs.com/png.latex?L_x\times%20D) matrix, where ![L_x](https://latex.codecogs.com/png.latex?L_x) is the length of the source sequence, and D is the dimension of latent variables. For the Transformer decoder to predict target tokens that have a length longer or shorter than ![L_x](https://latex.codecogs.com/png.latex?L_x), we need a funtion to adjust the length of latent variables, just like this:
 
<p align="center">
<img src="https://latex.codecogs.com/png.latex?\bar%20z%20=%20\mathrm{LengthTransform}(z,L_y)" />
</p>

As a result, ![z bar](https://latex.codecogs.com/png.latex?\bar%20z) will be a ![L_y by D](https://latex.codecogs.com/png.latex?L_y\times%20D) matrix. The implementation of this length transforming function can be found in [TBD]. 

## Install Package Dependency

The code depends on PyTorch, **torchtext** for data loading,
 **nmtlab** for Transformer modules and **horovod** for multi-gpu training.

Note that although you can train the model on a single GPU, but for a large dataset such as WMT14, the training takes a lot of time without multi-gpu support. We recommend you to get 4 ~ 8 GPUs for this task.

We recommend installing with conda.

-1. (If you don't have conda) Download and Install Miniconda for Python 3

```
mkdir ~/apps; cd ~/apps
wget  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Reload the bash/zsh and run `python` to check it's using the python in Miniconda.

-2. Install pytorch following https://pytorch.org/get-started/locally/

-3. (Only for multi-gpu training) Install horovod following https://github.com/horovod/horovod#install

```
mkdir ~/apps; cd ~/apps
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz
tar xzvf openmpi-4.0.1.tar.gz
cd openmpi-4.0.1
# Suppose you have Miniconda3 in your home directory
./configure --prefix=$HOME/miniconda3 --disable-mca-dso
make -j 8
make install
```

Check whether the openmpi is correctly installed by running `mpirun`. Then install horovod with:

```
conda install -y gxx_linux-64
# If you don't have NCCL
pip install horovod
# If you have NCCL in /usr/local/nccl
HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=/usr/local/nccl pip install horovod
```

Check horovod by running `horovodrun`.

-5. Run `pip install torchtext nmtlab`

-6. Clone this github repo, run 

```
cd ~/
git clone https://github.com/zomux/lanmt
cd lanmt
```

## Download pre-processed WMT14 dataset

We pre-processed the WMT14 dataset with *sentencepiece* and make the vocabulary size 32k for both source and target sides. For knowledge distllation, we use a baseline Transformer model to generate translations for the whole dataset. To save time, you can just download the pre-processed dataset from our link.

-1. Create `mydata` folder if it's not there

```
mkdir mydata
cd mydata
```

-2. Download pre-processed WMT14 dataset from https://drive.google.com/file/d/16w3ZmxbiRzRG8vtBh26am-GUldHYYvLv/view .
After download, uncompress the dataset in side `mydata` folder.

```
./gdown.pl https://drive.google.com/file/d/16w3ZmxbiRzRG8vtBh26am-GUldHYYvLv/view lanmt_wmt14.tgz
tar xzvf lanmt_wmt14.tgz
```

## Train the model

-1. Go back to `lanmt` folder

-2. (Single GPU) Run this command:
```
# If you have 16GB GPU memory
python run.py --opt_dtok wmt14_e
nde --opt_batchtokens 4092 --opt_distill --opt_annealbudget --train
# If you have 32GB GPU memory
python run.py --opt_dtok wmt14_e
nde --opt_batchtokens 8192 --opt_distill --opt_annealbudget --train
```

-2. (Multi-GPU) Run this command if you have 8 GPUs:
```
# If you have 16GB GPU memory
horovodrun -np 8 -H localhost:8 python run.py --opt_dtok wmt14_e
nde --opt_batchtokens 4096 --opt_distill --opt_annealbudget --train
# If you have 32GB GPU memory
horovodrun -np 8 -H localhost:8 python run.py --opt_dtok wmt14_e
nde --opt_batchtokens 8192 --opt_distill --opt_annealbudget --train
```

There are some options you can use for training the model:

``--opt_batchtokens`` specifies the number of tokens in a batch

``--opt_distill`` enabling knowledge distillation, which means the model will predict the output of a teacher Transformer

``--opt_annealbudget`` enabling annealing of the budget of KL divergence

In our experiments, we train the model with 8 GPUs, putting 8192 tokens in each batch. If the script is successfully launched, you will see outputs like this:

```
[nmtlab] Training TreeAutoEncoder with 74 parameters
[nmtlab] with Adagrad and SimpleScheduler
[nmtlab] Training data has 773 batches
[nmtlab] Running with 8 GPUs (Tesla V100-SXM2-32GB)
[valid] loss=6.62 label_accuracy=0.00 * (epoch 1, step 1)
[valid] loss=1.43 label_accuracy=0.54 * (epoch 1, step 194)
...
[nmtlab] Ending epoch 1, spent 2 minutes
...
[valid] loss=0.47 label_accuracy=0.86 * (epoch 12, step 14687)
...
```

## Inference

To generate translations and measure the decoding time, simply run

```
python run.py --opt_dtok wmt14_e
nde --opt_batchtokens 8192 --opt_distill --opt_annealbudget --test
```



Then, let's try to refine the latent variables with deterministic inference for only one step

```
python run.py --opt_dtok wmt14_e
nde --opt_batchtokens 8192 --opt_distill --opt_annealbudget --test --opt_Trefine_steps 1
```



We can also sample multiple latent variables from the prior, getting multiple candidate translations then use an autoregressive Transformer model to rescore them, you can do this by

```
python run.py --opt_dtok wmt14_e
nde --opt_batchtokens 8192 --opt_distill --opt_annealbudget --test --opt_Trefine_steps 1 --opt_Tlatent_search --opt_Tteacher_rescore
```



## Summary of  results



| Options | BLEU |
| ------- | ---- |
|         |      |

