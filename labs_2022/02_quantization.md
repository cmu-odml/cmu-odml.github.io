Lab 2: Quantization
===
This lab is a opportunity to explore different quantization strategies and settings, with the goal of becoming familiar with post-training quantization (PTQ) in PyTorch, and thinking about how and when quantization can be a useful strategy for model compression. 

You will be looking at how three properties vary as a result of model quantization:
1. Model size (in memory)
2. Model accuracy
3. Model latency (small vs large batch)

The techniques we are exploring are:
1. Dynamic Quantization
2. Static Quantization 

Values in our models will be quantized from `float32` to `float16` and `qint8`.
Additionally, all results will be reported both for SST-2 and MNIST. Due to hardware differences, several of the results in this lab are unlikely to directly transfer to other people's settings. 

Hardware
----
You are welcome to use any hardware to which you have acccess for this assignment as long as you clearly include that information in your report, but *please note that PyTorch Quantization 
is not currently supported for the Apple M series chips.* So if you have a newer Macbook (November 2020 or later), you will need to use Colab or another compute resource. 
One way to check whether PyTorch quantization is supported for your hardware is to run the following:
```
import torch.quantization
torch.backends.quantized.engine
```
If the result is not `fbgemm` then your hardware may not be supported.

**REPORT:** Document what hardware you are using for this assignment: 
- CPU
- RAM 
- Graphics or ML accelerator, in as much detail as is available to you.

Data, Models and Evaluation
----
Data, models and evaluation for this assignment will build off the code you wrote for Lab 1. You are encouraged (and expected) to re-use model
and evaluation code from Lab 1 to complete this assignment. In particular, you may find the following components of your Lab 1 useful for Lab 2:
- Preprocessing code for both MNIST and SST-2
- Model class definition
- Parameter count and latency timing code
- Training and evaluation code
  - You will run inference many more times than training during this lab, so the evaluation function is especially important. If you have not already, you may wish to integrate latency measurement functionality in your evaluation function. An example eval function might take in a model, dataloader and return both accuracy and total time used for inference.

**Models:**
For the models in this assignment you will experiment with variants of the feed-forward networks with ReLU activations that you implemented in Lab 1. You will again experiment with a computer vision model trained on MNIST, and a text sentiment analysis model trained on SST-2. You should only train **four** models, described as follows:

1. Lab 1 MNIST
2. Lab 1 SST-2
3. Lab 2 MNIST (see next table)
4. Lab 2 SST-2 (see second table below)

The Lab 1 models are the models trained with the default hyperparameters from Lab 1, and no transformations of the input (vocab size, image resolution):

| hyperparameter | value |
| --- | --- |
| learning rate | 0.001 |
| batch size | 100 |
| hidden size  | 512 | 
| # hidden layers | 1 | 
| training epochs | 2 |

### MNIST
Here are the model and training hyperparameters you should use for the Lab 2 MNIST model:
| hyperparameter  | value |
| --------------- | ----- |
| learning rate   | 0.001 |
| batch size      | 64    |
| hidden size     | 1024  | 
| # hidden layers | 2     |
| training epochs | 2     |

You should also crop the input to 20x20 pixels at the center of the image.
Train and report its accuracy on the MNIST test set. 
You should be able to get around 95% accuracy.

### SST-2
Here are the hyperparameters you should use for the Lab 2 SST-2 model: 

| hyperparameter  | value |
| --------------- | ----- |
| learning rate   | 0.001 |
| batch size      | 64    |
| hidden size     | 256   | 
| # hidden layers | 3     |
| training epochs | 2     |

For SST, you should threshold your vocabulary to the 5000 most frequent words in the vocabulary. 
Train this model and report its accuracy on the SST development set. You should be able to get about 80% accuracy.

**Evaluation:**

In this lab you will be evaluating models in terms of model size and inference latency. For inference latency report the average
**inference** time and standard deviation measured over 5 runs. For model size, you should report both parameter counts (using your Lab 1 code) and size on disk (in MB), which can be measured using this function:

```
import os

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size
```
You can then compare model sizes:

```
f=print_size_of_model(ffn_mnist,"fp32")
q=print_size_of_model(quantized_mnist,"int8")
print("{0:.2f} times smaller".format(f/q))
```

**REPORT:**
Evaluate and report the inference latency of both models for batch sizes in [1, 64]. 
Report the size of all four models.
This provides your baselines for comparison to quantization. You should fill out all the cells containing ? in the tables below.

#### MNIST
| Model | Q  | dtype     | Size (MB) | (Params) | Accuracy | Latency B1 | B64 |
| ----- | -- | -----     | ---- | -------- | -------- | ---------- | --- |
| Lab 1 | -- | `float32` |   ?   |     ?     |     ?       |  ?  |  ?    |
| Lab 2 | -- | `float32` |   ?   |    ?      |       ?     |  ?   | ?     |

#### SST
| Model | Q  | dtype     | Size (MB) | (Params) | Accuracy | Latency B1 | B64 |
| ----- | -- | -----     | ---- | -------- | -------- | ---------- | --- |
| Lab 1 | -- | `float32` |  ?    |   ?       |  ?          |  ?   |  ?    |
| Lab 2 | -- | `float32` |   ?   |    ?      |    ?        |  ?   |  ?    |



Dynamic quantization in PyTorch  
----
First, you will apply dynamic quantization to your models. Follow the [PyTorch quantization tutorial](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html) 
to implement PyTorch dynamic quantization for your models.

Quantize to both `float16` and `qint8` precision, and report your results. 

**REPORT:**
Copy values from above and expand. You should fill out all the cells containing ? in the tables below, with the other cells coming from previous tables.

| Model | Q  | dtype     | Size (MB) | (Params) | Accuracy | Latency B1 | B64 |
| ----- | -- | -----     | ---- | -------- | -------- | ---------- | --- |
| Lab 1 | -- | `float32` |      |          |            |     |      |
| Lab 1 | D  | `float16` |  ?   |    ?     |      ?     |  ?  |   ?  |
| Lab 1 | D  | `qint8`   |  ?   |    ?     |      ?     |  ?  |   ?  |
| Lab 2 | -- | `float32` |      |          |            |     |      |
| Lab 2 | D  | `float16` |  ?   |    ?     |      ?     |  ?  |   ?  |
| Lab 2 | D  | `qint8`   |  ?   |    ?     |      ?     |  ?  |   ?  |

**Discussion:**
Describe and discuss your observations, reported in the table. 
You might analyze differences across quantization precisions, models, tasks, or something else that you noticed. 
Choose **two** axes of interest and discuss each in more detail. They need not be the same two as above.

Static quantization in PyTorch
----
Next, you'll perform static quantization using [prepare_fx](https://pytorch.org/docs/stable/generated/torch.quantization.quantize_fx.prepare_fx.html) from [torch.quantization](https://pytorch.org/docs/stable/quantization-support.html).  

**REPORT:**
Copy values from above and expand two new lines:
| Model | Q  | dtype     | Size (MB) | (Params) | Accuracy | Latency B1 | B64 |
| ----- | -- | -----     | ---- | -------- | -------- | ---------- | --- |
| Lab 1 | -- | `float32` |      |          |            |     |      |
| Lab 1 | D  | `float16` |      |          |            |     |      |
| Lab 1 | D  | `qint8`   |      |          |            |     |      |
| Lab 1 | S  | `qint8`   |  ?   |    ?     |     ?      |  ?  |   ?  |
| Lab 2 | -- | `float32` |      |          |            |     |      |
| Lab 2 | D  | `float16` |      |          |            |     |      |
| Lab 2 | D  | `qint8`   |      |          |            |     |      |
| Lab 2 | S  | `qint8`   |  ?   |    ?     |     ?      |  ?  |   ?  |


**Discussion:**
Describe and discuss your observations, reported in the table. 
You might analyze differences across quantization precisions, models, tasks, or something else that you noticed. 
Choose **two** axes of interest and discuss each in more detail. They need not be the same two as above.

Overall discussion and future work
----
1. **Describe one observation you made** while completing this assignment that you found most interesting or surprising. 
2. Now, **pose a follow-up question related to this observation** (a research question or hypothesis) that is unanswered by the work in this assignment, 
and; 
3. **Describe in detail an experiment of set of experiments that you might perform** in order to answer that question. You do not need to run the experiments, just describe how you might go about answering or at least helping to elicidate an answer to the question. 

Extra Credit: 
----
This assignment includes two opportunities for extra credit.

#### 1. Benchmarking mixed precision training [0.5 points]
Add an additional comparison to your results table using [automatic mixed precision training](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html).
This requires access to appropriate hardware that supports mixed-precision training (either a personal or lab GPU or via Colab).

This should add 4 lines to your table: one for each model and dataset. You don't need to use the same hardware here as above, just clearly specify either way.

#### 2. Module-by-module sensitivity analysis [2 points]
[Start here](https://pytorch.org/blog/quantization-in-practice/#sensitivity-analysis). 
1. Analyze accuracy and inference time in both Lab 2 models from quantizing just one module or layer at a time. Your results will depend in part on the structure of your model; looping through the `named_modules` as in the documentation code will include modules as well as "leaf" layers.
2. Analyze accuracy and inference time in both Lab 2 models from quantizing *all but one* layer at a time. *Hint: you should loop through only modules that do not have child modules themselves*


Grading and submission (10 points + 2.5 extra credit)
----
Submit your answers to all the above questions to Canvas as a write-up in pdf format. This assignment is worth 10 points 
(or 10% of your final grade for the class), distributed as follows: 
- **Submission [2 points]:** Assignment is submitted on time.
- **Basic requirements [5 points]:** Answers to all the questions (including all requested tables) are included in the write-up. 
- **Report [2 points]:** Report is well-written and organized with minimal typos and formatting mistakes, and answers to all requested questions are easy to find. Tables are readable and well-labeled. Your results should be easily reproducible from the details included in your report.
- **Thoughtful discussion [1 points]:** The discussion of results in your write-up is thoughtful: it indicates meaningful engagement with the exercise and it is clear that you put effort into thinking through and explaining your results.
- **Extra Credit [2.5 points]:** See above for description of possible extra credit.
