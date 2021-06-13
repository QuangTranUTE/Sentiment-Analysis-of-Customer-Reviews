# Sentiment Analysis of Customer Reviews

Thanks Bao for the data

## What is this?
This repository contains code for training a rcurrent neural network with gated recurrent unit (GRU) cells. The model is used to analyse customer reviews from e-commerce websites. A model trained with about 100.000 product reviews (in Vietnamese) can be found in `models` folder.

## How to use it
#### *1. Using it as a product review analyser* 
Simply go to this [**Google Colab**](https://colab.research.google.com/drive/1cYNHmXSNTxkkt5wJsWcNx9wf35fkFZMW?usp=sharing), then click **Run cell** (Ctrl-Enter), input a review in Vietnamese and click **Analyze** 

![Demo using the translator on Colab](/resources/demo.PNG "Hope you enjoy it!") 


#### *2. Using the training code to create your own translator*
Steps to do:

    1. Prepare you data (see "datasets" for more details)
    2. (Optional) Customize the model in "train.py"
    3. Run "train.py" 

You can easily customize the model by changing hyperparameters put at the beginning of code parts (marked with comments `NOTE: HYPERPARAM`) (see `train.py`).  

Please be aware that the training process may take days to finish, depending on your customized model and your computer.  

After training, you can deploy your model on, for example, a Colab as I have done above.  

## Trained model information
The model was trained with the [**OPUS TED2020-v1 en-vi text**](https://opus.nlpl.eu/TED2020-v1.php) data with more than 300.000 pairs of text sequences (see `datasets` folder for details). 

The current model (in `train.py`) is a simple encoder-decoder with 4-GRU-layer encoder and decoder. Due to the lack of resources, *attention mechanism* (with *bidirectional RNNs*) have not been used.  

In the future, *beam search* and *random translation* may also be added to improve performance of the model.  

***Note for ones who want to implement attention mechanism:*** due to some bugs in the current Tensorflow (version 2.5.0) and Tensorflow Addons (version 0.13 as I have tried), we cannot implement attention mechanism in eager mode, to the best of my knowledge. So at the moment, you have to use the subclassing API if you wish to get attention mechanism. There are online tutorials with code can help you with that.

## Training data infomation
Data used to train the model in the repository can be downloaded [here](https://drive.google.com/file/d/1AiUt7TuIUcVLb3M_iM99yGhJTtuhOB_x/view?usp=sharing). Training data is the en-vi language pair of the [OPUS TED2020v1 data](https://opus.nlpl.eu/TED2020-v1.php).   

**Copyright**  

The training data I have used are taken from the OPUS corpus:  

> Website: http://opus.nlpl.eu
> 
> Please cite the following article if you use any part of the corpus in your own work: J. Tiedemann, 2012, Parallel Data, Tools and Interfaces in OPUS. In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012)
> 
> This dataset contains a crawl of nearly 4000 TED and TED-X transcripts from July 2020. The transcripts have been translated by a global community of volunteers to more than 100 languages. The parallel corpus is available from https://www.ted.com/participate/translate






