# Deep Learning and NLP in Keras (Text Classification)

Collection of Deep Learning Text Classification Models in Keras; Includes a GPU tutorial. All gpu models contained within the repo were trained with 4 v100's using the AWS p3.8xlarge instance and the AWS deep learning AMI. 

***Disclaimer: The dataset for this repository contains text that may be considered profane, vulgar, or offensive.***

## Contents
- [News](#news)
- [Overview](#contents-summary)
- [Competition and Inspiration](#competition-and-inspiration)
- [Evaluation and Benchmarking](#evaluation-and-benchmarking)
- [Data Overview](#data-overview)
- [Models Covered](#models-covered)
- [License](#license)

## News
- 04/14/2018: tutorial v1.0 complete 
- 04/14/2018: utils added to models direction
- 04/14/2018: utils tutorial complete
- 04/21/2018: tutorial v1.2 complete
- 04/21/2018: added BLSTM-2DCNN and BGRU-2DCNN models

## Contents Summary
- Collection of Deep Learning Text Classification Models and Benchmarks *(eta: soon)*
- Python directory with reusable utils and models
	- Py file with reusable utils
	- Py file with models
- A comprehensive and in-depth tutorial on NLP (classifying toxic comments), Deep Learning, and Keras. Using:
	- Keras
	- Text Classification Improved by Integrating Bidirectional LSTM with Two-dimensional Max Pooling (Peng Zhou, et al. 2016) https://arxiv.org/pdf/1611.06639.pdf
	- Toxic Comment Classification Challenge (Kaggle Competition)
	- A little humor
- Requirements *(eta: soon)*
- Conda Environment yml *(eta: soon)*

## Competition and Inspiration
Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments. The Kaggle Toxic Comment Classification Challenge sponsored by the Conversation AI team sets out to discover and apply machine learning to identify toxic comments. This (potentially) will allow platforms to identify toxic comments and to successfully fascilitate discussions at scale. 

In this competition, you’re challenged to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s current models. You’ll be using a dataset of comments from Wikipedia’s talk page edits. Improvements to the current model will hopefully help online discussion become more productive and respectful.

Source: Toxic Comment Classification Challenge, https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/  

## Evaluation and Benchmarking
Submissions are evaluated on the mean column-wise ROC AUC. In other words, the score is the average of the individual AUCs of each predicted column.

All benchmarks will be performed on the same 10% of hold out data. A model trained on the entire dataset will be used to submit late submissions to Kaggle. The results will be reported inside the notebook.

## Data Overview
We are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

We must create a model which predicts a probability of each type of **non-exclusive** toxicity for each comment.

You may obtain the competition data here: 
	
- https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

You may obtain the the pretrained word embeddings (glove.42B.300d) here:

- https://nlp.stanford.edu/projects/glove/

## Models Covered
- BLSTM-2DCNN - https://arxiv.org/pdf/1611.06639.pdf
- BGRU-2DCNN - *modification of:* https://arxiv.org/pdf/1611.06639.pdf
- More to come *(eta: soon)*

## Hardware
INSTANCE: p3.8xlarge

- https://aws.amazon.com/ec2/instance-types/p3/

AMI: Ubuntu CUDA9 DLAMI with MXNet/TF/Caffe2

-   https://docs.aws.amazon.com/mxnet/latest/dg/CUDA9_Ubuntu1.html

## LICENSE
MIT License

Copyright (c) 2018 John Daniel Paletto

