Lab 1: Benchmarking Efficiency in Feed-Forward Neural Networks.
===
In this lab you will build a basic feed-forward neural network for classification in PyTorch, and train and evaluate that model in terms of efficiency and classification accuracy on simple
language and vision tasks. The goals of this exercise are: 
 1. Proficiency training and evaluating basic feed-forward neural network architectures for language and computer vision classifcation tasks in PyTorch; 
 2. Implement basic efficiency benchmarks: Latency, parameter count, and FLOPs.
 3. Experiment with varying model size, depth, and input resolution and analyze how that impacts efficiency vs. accuracy.

Data
----

### Language 
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

### Vision
For a vision task we will use the [MNIST dataset of handwritten digits](https://en.wikipedia.org/wiki/MNIST_database). 
Examples in this dataset consist of 28x28 greyscale images with pixel values between 0 and 255, labeled with digits 0-9.
The data files we will use for this class are available [here](https://pjreddie.com/projects/mnist-in-csv/). 
The files are formatted as follows, with one example per line:
```
label, pix-11, pix-12, pix-13, ...
```
You can read more about the dataset in the original paper [here](https://papers.nips.cc/paper/1989/hash/53c3bce66e43be4f209556518c2fcb54-Abstract.html).
The provided split contains 60,000 training examples and 10,000 test examples.

Features
----

### Language 
For the language task, the tokens of text need to be converted into something consumable by a neural network. 
Today, we will use a bag-of-words representation.
A bag-of-words representation is a fixed-length vector of length $|V|$ where $|V|$ is the size of the vocabulary $V$.
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

### Vision
The raw pixel values will work as input for now.

Model and Evaluation
----
Build a single-layer feed-forward network with ReLU activations and cross-entropy loss using PyTorch, and train it on the vision and language tasks. You can experiment with a few different settings of hyperparameters to find ones that work reasonably well for each modality. To start, you should be able to get an accuracy of about 97% using the following hyperparameters for MNIST using the Adam optimizer:

| hyperparameter | value |
| --- | --- |
| learning rate | 0.001 |
| batch size | 100 |
| hidden size  | 512 | 
| epochs | 2 |

Compute classification accuracy for each task.
1. What accuracy do you obtain on each task, and using what hyperparameters? This will be your "base model."

Now, implement three methods for benchmarking the efficiency of this model:
- **Latency:** Measure training time and inference latency. You can simply use `time.time()` in Python around your training loop to get an estimate of training time. For inference latency, measure the average time it takes to classify each example. You should run this a few times, and throw away the first measurement to allow for initial warmup (e.g. caching, etc.) 
- **Parameter count:** Write a function to compute the number of parameters in your model. This should be a general-purpose function that can be run on any PyTorch model; rather than estimate this in closed form, you should loop over the tensors of parameters in the model and sum their sizes.
- **FLOPs:** Write a function to compute the number of floating-point operations that need to be performed in order to do inference on one example in your model. This one does not have to be general-purpose, it should be specific to feed-forward networks.

2. What is the exact hardware that you are using to benchmark your code? Report the CPU and RAM, in as much detail as is available from your operating system.
3. Report the average training time and inference latency of your model. Is the variance high or low? Did you notice any outliers? Was the first iteration of inference slower than the others? Try to explain any phenomena you note.
4. Report the parameter count of your model. Does this align with your expectations? Why or why not? 
5. Report the number of FLOPs that your model requires to perform inference.


Varying depth and width
----
Now, try varying the depth and width of your base model. How do depth and width trade off accuracy and efficiency?

6. Train and evaluate your model in both modalities using a variety of depths. Report the parameter counts, FLOPs, and latency, and generate three plots:
    - FLOPs on the x axis and accuracy on the y axis
    - Latency on the x axis and accuracy on the y axis
    - FLOPs on the x axis and latency on the y axis
   
    Discuss your results.
7. Train and evaluate your model in both modalities using a variety of widths. Make sure you try both going narrower than the input, and wider than the input, if memory permits. Report the parameter counts, and generate the same three plots as in (6), and discuss.

Varying the input size
----
Now, try varying the input size (input resolution) of your base model. 
- For vision, you can do this by *downsampling* the input. You can find a list of transformations [here](https://pytorch.org/vision/stable/transforms.html).
- For language, you can do this by reducing the vocabulary size.

8. **Vision:** Downsample by resizing the image to a smaller size, and experiment with a few different downsampling rates. Generate the same three plots as above. 
9. Repeat this for a different transformation, such as cropping. How does this different transformation compare to resizing?
10. **Language:** Downsample by reducing the vocabulary to the top $k$ most frequent words, for a few different values of $k$. Generate the same three plots.
11. Repeat this using a different method for selecting words, for example you may choose to remove [stopwords](https://gist.github.com/sebleier/554280). Be creative. How does this different transformation compare to keeping the most frequent words?

Putting it all together
----
Now, experiment with different settings combining different input sizes with different depths and widths. 

12. Generate the same three plots as above, reporting the results of your experiments in each modality. Do you notice any interesting trends? What combination of input resolution, depth, and width seems to correspond to the best trade-off between accuracy and efficiency? Do you notice differences between latency and FLOPs? Explain your experimental process and discuss your results.

Grading and submission (10 points)
----
Submit your answers to all the above questions to Canvas as a write-up in pdf format. This assignment is worth 10 points, graded as follows: 
- **Submission [2 points]:** Assignment is submitted on time.
- **Basic requirements [5 points]:** Answers to all the questions (including all requested plots) are included in the write-up.
- **Report [2 points]:** Report is well-written and organized with minimal typos and formatting mistakes, and answers to all requested questions are easy to find. Plots are readable and well-labeled. Your results should be easily reproducible from the details included in your report.
- **Thoughtful discussion [1 point]:** The discussion of results in your write-up is thoughtful: it indicates meaningful engagement with the exercise and it is clear that you put effort into thinking through and explaining your results.
