# Lab 1: Benchmarking Efficiency in Feed-Forward Neural Networks.

## 1. Assignment Objectives

In this lab you will build a basic feed-forward neural network for classification in PyTorch on CPU, and train and evaluate that model in terms of efficiency and classification accuracy on a simple language or vision task. The goals of this exercise are: 

 1. Proficiency training and evaluating basic feed-forward neural network architectures for language or computer vision classifcation tasks in PyTorch.
 2. Implement methods for benchmark basic efficiency metrics of: Latency, parameter count, and FLOPs.
 3. Experiment with varying model size, depth, and input resolution; and analyze the impacts on efficiency and accuracy.

We recommend installing [PyTorch](https://pytorch.org/get-started/locally/) using a Python package manager such as [Conda or Mamba](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).

*Note*: This is an individual lab to be done on your personal laptop.

**DUE: 11:59 PM EDT on September 10** 


## 2. Model Training, Evaluation and Efficiency Benchmarking

### 2.1 Datasets

For this assignment, select **one** of the following tasks as relevant to your interests or project.

#### Language: Sentiment Classification
For a language task we will use the [Stanford Sentiment Treebank (SST-2)](https://huggingface.co/datasets/sst2) dataset for sentiment classification. 
Examples in this dataset consist of tokenized English text labeled with binary categories indicating the binary sentiment (positive/negative) of the sentence. 
The data files we will use for this class are available [here](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip).

The files are formatted as follows, with one example per line:
```
tok1 tok2 tok3 ...  label
```
Note that the label is separated from the tokens by a tab character.
The provided split contains 67,350 training examples, 873 development examples and 1822 test examples.
You can read more about the dataset in the original paper available [here](https://www.aclweb.org/anthology/D13-1170). 


#### Computer Vision: MNIST

For a vision task we will use the [MNIST dataset of handwritten digits](https://en.wikipedia.org/wiki/MNIST_database). 
Examples in this dataset consist of 28x28 greyscale images with pixel values between 0 and 255, labeled with digits 0-9.
The data files we will use for this class are available [here](https://pjreddie.com/projects/mnist-in-csv/). 
The files are formatted as follows, with one example per line:
```
label, pix-11, pix-12, pix-13, ...
```
You can read more about the dataset in the original paper [here](https://papers.nips.cc/paper/1989/hash/53c3bce66e43be4f209556518c2fcb54-Abstract.html).
The provided split contains 60,000 training examples and 10,000 test examples.

### 2.2 Input Features

For processing, it is necessary to convert raw input data into format that can be utilized by a neural network (i.e. namely finite dimensional vectors) which will be sampled for mini-batch training. See [PyTorch Datasets and Dataloaders.](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

#### Language: Bag of Words
For the language task, text tokens will be converted a bag-of-words representation. A bag-of-words representation is a one-hot, fixed-length vector of length $|V|$ where each element in the vector corrresponds to a unique token in the vocabulary $V$.

To start, you can define the vocabulary as all unique words occurring in the training data.
For a given example, its bag-of-words vector will consist of a count $c_i$ of the number of times word $i$ occurred in the example text.
For example, consider $V$ = {dog, cat, lizard, green, beans, bean, sausage, the, my, likes, but} and the example sentence: 

```
my dog likes green beans but my cat likes sausage
```
Then the corresponding BoW vector would look like:

| dog | cat | lizard | green | beans | bean | sausage | the | my | likes | but |
| --- | --- | ------ | ----- | ----- | ---- | ------- | --- | -- | ----- | --- |
| 1   | 1   | 0      | 1     | 1     | 0    | 1       | 0   | 2  | 2     | 1   |

#### Vision: Pixels

For the computer vision task, the model can operate directly over the raw pixel inputs. However, it is common to preprocess inputs for stability in model convergence by normalizing values to have mean 0 and standard deviation 1.

This can be done by computing the mean and standard deviation over all instances in the training data, and update the values as:
$$new = \frac{old - mean}{stdev}$$

### 2.3 Model Architecture and Training

Construct a feed-forward network with ReLU activations and cross-entropy loss using PyTorch; and train it on your choice of SST-2 and MNIST classification tasks. The resulting "base model" will be reused for subsequent sections.

You can experiment with different settings of hyperparameters to determine appropriate values for each modality. To start, you should be able to get an accuracy of about 97% using the following hyperparameters for MNIST using the Adam optimizer.

| Hyperparameter | Value |
| --- | --- |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| hidden Layers | 2 |
| Hidden Size  | 1024 (MNIST), 256 (SST) | 
| Epochs | 2 |

**Note**: The input and output layers are not considered hidden layers. 

### 2.4 Model Efficiency
Now, implement three methods for benchmarking the model efficiency:

1. **Latency:** Measure training time and inference latency. In a CPU-only environment, computational kernels are executed synchronously so the Python `time` library is sufficient for estimating wallclock time. 
   - For training latency, measure the required time to train each epoch and the total training time.
   - For inference latency, measure the average time it takes to classify each example. You should run this a few times, and throw away the first measurement to allow for initial warmup (e.g. caching, etc.) 

2.  **Parameter Count:**  Manually compute the number of trainable parameters in your model, and write a function that gathers the number of trainable parameters in PyTorch. 
    - Provide a closed form for Manually compute the number of trainable parameters
    - This should be a general-purpose function that can be run on any PyTorch model. Simply, iterate through the tensors for each model parameters in the model and sum their sizes. 
    

3. **Floating Point Operations (FLOPs):** Write a function to compute the number of floating-point operations for inference on one example with your model.
   -  This one does not have to be general-purpose, it should be specific to feed-forward networks
   -  Explain your approach. Answers will be graded on clarity and reasoning.


## 3. Questions 

### Base Model Efficiency

1. What is the accuracy of the model on the train split and dev splits? Report the resulting hyperparameters.

2. What is the experimental setting used to benchmark the model? Report the software (e.g. Python and PyTorch Version, Operating System, etc.) and hardware configurations (e.g. CPU, RAM, etc.) in as much detail as is available from your operating system.

3. Report the average training time per epoch and inference latency per example of your model. 
   * Is the variance high or low? Did you notice any outliers? Was the first iteration of inference slower than the others?
   * Explain any phenomena you note.

4. Report the parameter count of the model as manually calculated; and as determined by your code solution. Does this align with your expectations? Why or why not? 

5. Report the number of FLOPs that your model requires to perform single-batch inference.


### Varying depth and width

Now, try varying the depth and width of your base model. How do depth and width trade off accuracy and efficiency?

6. Train and evaluate your model using a variety of depths. Report the parameter counts, FLOPs, and latency, and generate three plots:
    - FLOPs on the x-axis and accuracy on the y-axis
    - FLOPs on the x-axis and latency on the y-axis
    - Latency on the x-axis and accuracy on the y-axis
   
    Discuss your results.

7. Train and evaluate your model using a variety of widths. Make sure you try both going narrower than the input, and wider than the input, if memory permits.
   -  Report the parameter counts, and generate the same three plots as in Question 6, and discuss.

### Varying the input size

Now, try varying the input size (input resolution) of your base model (see corresponding sections). 
 
8. **Vision:** Downsample by resizing the image to a smaller size, and experiment with a few different downsampling rates. 
    - Generate the same three plots as in Question 6.
    -  Repeat this for a [different transformation](https://pytorch.org/vision/stable/transforms.html), such as cropping. How does this different transformation compare to resizing?

9. **Language:** Downsample by reducing the vocabulary to the top $k$ most frequent words, for a few different values of $k$.
    - Generate the same three plots as in Question 6.
    - Repeat this using a different method for selecting words, for example you may choose to remove [stopwords](https://gist.github.com/sebleier/554280). Be creative. How does this different transformation compare to keeping the most frequent words?

### Putting it all together

Now, experiment with different settings combining different input sizes with different depths and widths. 

10. Generate the same three plots as above, reporting the results of your experiments. 
     - Do you notice any interesting trends? What combination of input resolution, depth, and width seems to correspond to the best trade-off between accuracy and efficiency? Does variation of input size or model size have a larger impact on efficiency or performance?  Do you notice differences between different efficiency metrics latency and FLOPs? 
     - Explain your experimental process and discuss your results.

## Grading and submission (10 points)

Submit your answers to all the above questions to Canvas as a write-up in pdf format. This assignment is worth 10 points, graded as follows: 

- **Submission [2 points]:** Assignment is submitted on time.

- **Basic requirements [5 points]:** Answers to all the questions (including all requested plots) are included in the write-up.

- **Report [2 points]:** Report is well-written and organized with minimal typos and formatting mistakes, and answers to all requested questions are easy to find. Plots are readable and well-labeled. Your results should be easily reproducible from the details included in your report.

- **Thoughtful discussion [1 point]:** The discussion of results in your write-up is thoughtful: it indicates meaningful engagement with the exercise and it is clear that you put effort into thinking through and explaining your results.

