# Sentiment Analysis of Customer Reviews

## What is this?
This repository contains code for training a rcurrent neural network with gated recurrent unit (GRU) cells. The model is used to analyse customer reviews from e-commerce websites. A model were trained using a dataset consisting of about 100.000 product reviews (in Vietnamese). See `models` folder for more info about the data.

## How to use it
#### *1. Using it as a product review analyser* 
Simply go to this [**Google Colab**](https://colab.research.google.com/drive/1cYNHmXSNTxkkt5wJsWcNx9wf35fkFZMW?usp=sharing) (or [**this**](https://colab.research.google.com/drive/1EJwhfZV4LYJLPKlPvTtBYTdcilKBbJ1G?usp=sharing) for Vietnamese version), then click **Run cell** (Ctrl-Enter), input a review in Vietnamese and click **Analyze**.

![Demo using the trained model on Colab](/resources/demo.PNG "Hope you enjoy it!") 


#### *2. Using `train.py` to create your own model*
Steps to do:

    1. Prepare you data (see "datasets" for details)
    2. (Optional) Customize the model in "train.py"
    3. Run "train.py" 

You can easily customize the model by changing hyperparameters put at the beginning of code parts (marked with comments `NOTE: HYPERPARAM`) (see `train.py`).  

After training, you can deploy your model on, for example, a Colab as I have done above.  

## Brief info about the training data and trained model 
The trained model (in `models`) is a simple RNN with 5 GRU layers. It was trained with about 100.000 reviews crawled from Vietnamese e-commerce sites (see `datasets` folder for details). 

The model ouputs probabilities of the review to belong to classes:

        0: Dislike, unsatisfied with the purchase
        1: Like, satisfied with the purchase
        2: Neutral, unclear

The data were crawled (from Vietname e-commerce websites in 2021) and labeled by *Trần Gia Bảo, Trần Thị Tâm Nguyên, Hoàng Thị Cẩm Tú,* and *Uông Thị Thanh Thủy*. Many thanks for their hard work!            






