LaNMT: Latent-variable Non-autoregressive Neural Machine Translation with Deterministic Inference
===

>>> Update: this paper is accepted by AAAI 2020.

LaNMT implements a latent-variable framework for non-autoregressive neural machine translation. As you can guess from the code, it's has a simple architecture but powerful performance. For the details of this model, you can check our paper on Arxiv https://arxiv.org/abs/1908.07181 . To cite the paper:

```
@article{Shu2020LaNMT,
  title={Latent-Variable Non-Autoregressive Neural Machine Translation with Deterministic Inference Using a Delta Posterior},
  author={Raphael Shu and Jason Lee and Hideki Nakayama and Kyunghyun Cho},
  journal={AAAI},
  year={2020}
}
```
#### What is non-autoregressive neural machine translation?

In conventional neural machine translation modes, the decoder side is a language model. That means the model generate a single word in each time step. So you have to compute the neural network by ![N](https://latex.codecogs.com/png.latex?\fn_cs%20N) times in order to get a translation of ![N](https://latex.codecogs.com/png.latex?\fn_cs%20N) words. See the illustration below:

<p align="center">
<img src="https://i.imgur.com/0JZPqFH.png" width="800px"/>
</p> 

Such models can't fully exploit the parallelizability of GPU as you have to wait preceeding words to be generated to find the next word. In constrast, non-autoregressive models generate all target words in just one run of neural computation. As all target tokens are predicted simutaneously, the translation speed can be much faster.

#### Our model

We learn a set of continuous latent variables ![z](https://latex.codecogs.com/png.latex?\fn_cs%20z) to capture the information and intra-word dependencies of the target tokens. Intuitively, if the model is perfectly trained and the target sequence can be fully reconstructed from the latent variables without error, then the translation problem becomes a problem of finding adequate ![z](https://latex.codecogs.com/png.latex?\fn_cs%20z). This is illustrated in the picture below, which shows the relations among ![x](https://latex.codecogs.com/png.latex?\fn_cs%20x), ![z](https://latex.codecogs.com/png.latex?\fn_cs%20z) and ![y](https://latex.codecogs.com/png.latex?\fn_cs%20y).

<p align="center">
<img src="https://i.imgur.com/qh7sPlB.png" width="400px"/>
</p> 

In practice, we force the latent variables to have very low dimensions such as 8. Obviously, handling things in a low-dimension countinuous space is easier comparing to a high-dimension discrete space.

Our model is trained by maximizing the following objective, which is a lower bound of log-likehood. We  call it *evidence lower bound* (ELBO). The first part is a reconstruction loss that makes sure you can predict target sequence from ![z](https://latex.codecogs.com/png.latex?\fn_cs%20z). The second part is a KL divergence, which makes the ![z](https://latex.codecogs.com/png.latex?\fn_cs%20z) more predictable given the source sequence.

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\fn_cs%20\log%20p(y|x)%20\geq%20\mathbb{E}_{z%20\sim%20q(z|x,y)}%20\Big[\log%20p(y|x,z,l_y)%20+%20\log%20p(l_y|z)\Big]%20-%20\mathrm{KL}\Big(q(z|x,y)||p(z|x)\Big)" />
</p>

Now for the parameterization, the model is implemented with the architecture in the picture below. Does it appear to be more complicated comparing to a standard Transformer? Well, you are now computing four probabilities instead of only ![p(y|x)](https://latex.codecogs.com/png.latex?\fn_cs%20p(y|x)). However, as the model is basically reusing the Transformer modules such as self-attention and cross-attention, it's still pretty easy to implement. 

<p align="center">
<img src="https://i.imgur.com/a3x9tni.png" width="400px"/>
</p> 

One thing special about this model is that the number of latent variables is always identical to the source tokens, as you can guess from the second figure in this post. As each ![z_i](https://latex.codecogs.com/png.latex?\fn_cs%20z_i) is a continuous vector, ![z](https://latex.codecogs.com/png.latex?\fn_cs%20z) is a ![L_x by D](https://latex.codecogs.com/png.latex?\fn_cs%20L_x\times%20D) matrix, where ![L_x](https://latex.codecogs.com/png.latex?\fn_cs%20L_x) is the length of the source sequence, and D is the dimension of latent variables. For the Transformer decoder to predict target tokens that have a length longer or shorter than ![L_x](https://latex.codecogs.com/png.latex?\fn_cs%20L_x), we need a funtion to adjust the length of latent variables, just like this:

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\fn_cs%20z^\prime%20=%20\mathrm{LengthTransform}(z,L_y)" />
</p>

As a result, ![z'](https://latex.codecogs.com/png.latex?\fn_cs%20z^\prime) will be a ![L_y by D](https://latex.codecogs.com/png.latex?\fn_cs%20L_y\times%20D) matrix. The implementation of this length transforming function can be found in `lib_lanmt_modules.py` (class LengthConverter) . 

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

## Download pre-processed WMT14 dataset and teacher Transformer models

We pre-processed the WMT14 dataset with *sentencepiece* and make the vocabulary size 32k for both source and target sides. For knowledge distllation, we use a baseline Transformer model to generate translations for the whole dataset. To save time, you can just download the pre-processed dataset from our link.

-1. Create `mydata` folder if it's not there

```
mkdir mydata
cd mydata
```

-2. Download pre-processed WMT14 dataset from https://drive.google.com/file/d/16w3ZmxbiRzRG8vtBh26am-GUldHYYvLv/view . After download, uncompress the dataset in side `mydata` folder.

```
./gdown.pl https://drive.google.com/file/d/16w3ZmxbiRzRG8vtBh26am-GUldHYYvLv/view lanmt_wmt14.tgz
tar xzvf lanmt_wmt14.tgz
```

-3. (Option) Download pre-processed ASPEC Ja-En dataset. Due to copyright issue, we only provide test dataset and extracted vocabularies

```
./gdown.pl https://drive.google.com/file/d/1PhjJS1-NycqbW-LRSLiAZLVvZ00Xh5GW/view lanmt_aspec.tgz
tar xzvf lanmt_aspec.tgz
```



-4. Download teacher Transformer models (735MB) for rescoring candidate translations when performing latent search.

```
./gdown.pl https://drive.google.com/file/d/1xB81cmSQ7l66zZjWPEBhoc4nzjgFSWZW/view lanmt_teacher_models.tgz
tar xzvf lanmt_teacher_models.tgz
```



## Model Training

Here, we start to train the non-autoregressive model. Note that if you don't have time and just want to play with pre-trained model, please jump to https://github.com/zomux/lanmt#use-our-pre-trained-models .

-1. Go back to `lanmt` folder

-2. (Single GPU) Run this command:
```
# If you have 16GB GPU memory
python run.py --opt_dtok wmt14_ende --opt_batchtokens 4092 --opt_distill --opt_annealbudget --train
# If you have 32GB GPU memory
python run.py --opt_dtok wmt14_ende --opt_batchtokens 8192 --opt_distill --opt_annealbudget --train
```

-2. (Multi-GPU) Run this command if you have 8 GPUs:

```
# If you have 16GB GPU memory
horovodrun -np 8 -H localhost:8 python run.py --opt_dtok wmt14_ende --opt_batchtokens 4096 \
--opt_distill --opt_annealbudget --train
# If you have 32GB GPU memory
horovodrun -np 8 -H localhost:8 python run.py --opt_dtok wmt14_ende --opt_batchtokens 8192 \
--opt_distill --opt_annealbudget --train
```


There are some options you can use for training the model:

``--opt_batchtokens`` specifies the number of tokens in a batch

``--opt_distill`` enabling knowledge distillation, which means the model will predict the output of a teacher Transformer

``--opt_annealbudget`` enabling annealing of the budget of KL divergence

In our experiments, we train the model with 8 GPUs, putting 8192 tokens in each batch. If the script is successfully launched, you will see outputs like this:

```
[nmtlab] Running with 8 GPUs (Tesla V100-SXM2-32GB)
[valid] len_loss=2.77 len_acc=0.12 loss=194.92 word_acc=0.16 KL_budget=1.00
kl=27.87 tok_kl=1.00 nll=164.28 * (epoch 1, step 471)
...
[valid] len_loss=1.57 len_acc=0.40 loss=69.53 word_acc=0.66 KL_budget=1.00 k
l=28.41 tok_kl=1.02 nll=39.55 * (epoch 1, step 3761)
[nmtlab] Ending epoch 1, spent 53 minutes
...
```

In the training log, `loss` showes the total loss value, `nll` shows the cross-entropy value, `kl` shows the KL divergence, `tok_kl` shows the average KL value for each token and `len_loss` and `len_acc` shows the loss and prediction accuracy of the length predictor.

After finishing the model training, we also find it helpful to fix the KL budget at zero, and finetune the model for only one epoch. You can do this by running

```
# Single GPU
python run.py --opt_dtok wmt14_ende --opt_batchtokens 4092 --opt_distill --opt_annealbudget \
--opt_finetune --train
# Multi-GPU
horovodrun -np 8 -H localhost:8 python run.py --opt_dtok wmt14_ende --opt_batchtokens 4096 \
--opt_distill --opt_annealbudget --opt_finetune --train
```

## Inference

To generate translations and measure the decoding time, simply run

```
python run.py --opt_dtok wmt14_ende --opt_batchtokens 8192 --opt_distill --opt_annealbudget \
--opt_finetune --test --evaluate
```

You will see the decoding time and evaluated BLEU scores at the end of lines. Then, let's try to refine the latent variables with deterministic inference for only one step

```
python run.py --opt_dtok wmt14_ende --opt_batchtokens 8192 --opt_distill --opt_annealbudget \
--opt_finetune --opt_Trefine_steps 1 --test --evaluate
```

We can also sample multiple latent variables from the prior, getting multiple candidate translations then use an autoregressive Transformer model to rescore them, you can do this by running

```
python run.py --opt_dtok wmt14_ende --opt_batchtokens 8192 --opt_distill --opt_annealbudget \
--opt_finetune --opt_Trefine_steps 1 --opt_Tlatent_search --opt_Tteacher_rescore --test --evaluate
```

With the `--evaluate` option, the script will evalaute the BLEU scores with sacrebleu. Once the script finishes you shall see the decoding time and BLEU scores like this

```
Average decoding time: 89ms, std: 22
BLEU = 25.166677019716257
```

## Use our pre-trained models

If you just want to test out the model and check the decoding speed and quality of translations, you can download our pre-trained models. By running the script with these models, you will get exactly the same BLEU scores as we reported in the paper.

-1. Download the pre-trained models (1GB)

```
cd mydata
./gdown.pl https://drive.google.com/file/d/1DcTHZYuhJhxxh0153qRx6BkBNHDK_f3b/view lanmt_pretrained_models.tgz
tar xzvf lanmt_pretrained_models.tgz
cd ..
```



-2. Translate using pre-trained models

```
# Lightning fast translation
python run.py --opt_dtok wmt14_ende --use_pretrain --test --evaluate
# With one refinement step
python run.py --opt_dtok wmt14_ende --use_pretrain --opt_Trefine_steps 1 --test --evaluate
# With latent search and teacher rescoring
python run.py --opt_dtok wmt14_ende --use_pretrain --opt_Trefine_steps 1 --opt_Tlatent_search --opt_Tteacher_rescore --test --evaluate
```



-3. (Option) Evaluate the pre-trained model on ASPEC Ja-En dataset

```
# Lightning fast translation
python run.py --opt_dtok aspec_jaen --use_pretrain --test --evaluate
# With one refinement step
python run.py --opt_dtok aspec_jaen --use_pretrain --opt_Trefine_steps 1 --test --evaluate
# With latent search and teacher rescoring
python run.py --opt_dtok aspec_jaen --use_pretrain --opt_Trefine_steps 1 --opt_Tlatent_search --opt_Tteacher_rescore --test --evaluate
```



## Summary of  results

| Dataset         | Options                                                      | BLEU  | Decode Time (avg/std) | Speedup |
| --------------- | ------------------------------------------------------------ | ----- | --------------------- | ------- |
| **WMT14 En-De** | Our baseline Transformer (beam size=3)                       | 26.10 | 602ms / 274           |         |
|                 | `--use_pretrain`                                             | 22.30 | 18ms / 4              | 33.4x   |
|                 | `--use_pretrain --opt_Trefine_steps 1`                       | 24.14 | 46ms / 4              | 13.0x   |
|                 | `--use_pretrain --opt_Trefine_steps 1 --opt_Tlatent_search`  | 25.01 | 67ms / 18             | 8.9x    |
|                 | `--use_pretrain --opt_Trefine_steps 1 --opt_Tlatent_search --opt_Tteacher_rescore` | 25.16 | 89ms / 22             | 6.7x    |
| **ASPEC Ja-En** | Our baseline Transformer (beam size=3)                       | 27.15 | 415ms / 159           |         |
|                 | `--use_pretrain`                                             | 25.28 | 21ms / 4              | 19.7x   |
|                 | `--use_pretrain --opt_Trefine_steps 1`                       | 27.53 | 47ms / 8              | 8.8x    |
|                 | `--use_pretrain --opt_Trefine_steps 1 --opt_Tlatent_search`  | 28.08 | 69ms / 18             | 6.0x    |
|                 | `--use_pretrain --opt_Trefine_steps 1 --opt_Tlatent_search --opt_Tteacher_rescore` | 28.26 | 93ms / 23             | 4.5x    |



## Troubleshooting

1. Training is slow

> Try to install horovod with nccl support. Training will be much faster with nccl for gradient synchronization.

## Todos

- [ ] Support half precision training
- [ ] Validation with BLEU criteria
- [ ] Update the distillation data with a new baseline model
