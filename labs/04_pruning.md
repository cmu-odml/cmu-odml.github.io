# Lab 4: Pruning

In this lab, you will explore different pruning strategies and trade-offs for both a simple network trained on MNIST/SST as well as your own model, device, and data. The main goals of this exercise are: 
    1. Learn about different pruning strategies and settings
    2. Implement and evaluate pruning on a simple model
    3. Implement and evaluate pruning on your own model, device, and data
    4. Compare and contrast different pruning strategies and settings

Throughout this lab, we will use the term **sparsity level** to refer to the percent of the original model's *prunable* weights (**weight parameters, excluding bias parameters**) that have been pruned.

> Note: If you are compute-resource-constrained (i.e. your personal laptop takes a really long time to perform a training run with the base model), you may adjust the hyperparameters in Exercise 1 to reduce computational burden -- e.g. training epochs, hidden size, but please clearly report what you did, try to keep the changes minimal, and be consistent throughout the assignment.

We require that you submit your report as a pdf. We **do not** want you to submit _all_ of your code (leaving us with 40 pages of code and outputs and delaying your grading). Please only show us necessary code snippets where explicitly requested. You are free to use this file as a starting point.


**DUE DATE: Nov 22, 2024**

## Preliminaries

Copy over relevant code for training MNIST or SST2 from Lab 3 to your codebase for this lab (but do not submit it for this question), including code for evaluation, *but do not train yet*! In particular, you will need to copy over the code for computing the model's:
- Accuracy
- Inference latency
- Disk size


| hyperparameter  | MNIST | SST2  |
| --------------- | ----- | ----- |
| learning rate   | 0.001 | 0.001 |
| batch size      | 64    | 64    |
| hidden size     | 1024  | 256   |
| # hidden layers | 2     | 3     |
| training epochs | 2     | 2     |

You should also crop the input to 20x20 pixels at the center of the image.

Recall from Lab 3 that you can measure model size on disk in this way:

```py
import os

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size
```
You can then compare model sizes:

```py
full=print_size_of_model(model1, "model1")
pruned=print_size_of_model(model2, "model2")
print("{0:.2f} times smaller".format(full/pruned))
```

## [1 point] Exercise 1: Base Model & Initial Training

### [0.5 points] 1. Hardware Specifications 

Share the hardware specifications and OS information of the machine(s) you will be using to run the experiments in this lab.

### [0.5 points] 2. Initialize, save and train your model

Initialize your SST/MNIST model. Before training, **SAVE** your model's initial (random) weights. You will use them later for iterative pruning. You can save the model's state dict to disk.

Now train the base model and report dev accuracy, inference latency, number of parameters, and space needed for storage (in MB) of `state_dict()` on disk. Fill in the following table with the results:

| Iteration | Sparsity (%) | Accuracy | Latency (s) | Disk Size (MB) | Parameters |
| --------- | ------------ | -------- | ----------- | -------------- | ---------- |
|      0    |      0.0%    |     ?    |       ?     |        ?       |     ?      |


Also take a look at your model's `named_parameters()` to get an idea of what your parameters are named as. You will need these for exercise 2 (no need to put in the table).


## [5.5 points] Exercise 2: Magnitude pruning on SST2/MNIST

### [0.5 points] 1. Global unstructured magnitude pruning

First, you will perform global, unstructured magnitude (L1) pruning on the trained MNIST/SST model to a sparsity level of **33%**. Prune just the weight parameters (not biases). 

You should be able to use the `global_unstructured` pruning method in the PyTorch prune module.
For usage examples, see the [PyTorch pruning tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html).


Show only the code you used to define the parameters to prune, and the code you used to prune the model. Example code for defining parameters to prune is shown below:

```py
params_to_prune = [
    (model.layers[0], 'weight'),
    (model.layers[1], 'weight'),
    (model.out, 'weight')
]
```

or

```py
params_to_prune = [
    (m[1], "weight") for m in model.named_modules() if len(list(m[1].children())) == 0
]
```

Take a look at your `model.named_parameters()` again, and your `model.named_buffers()`. (Just observe, no need to answer a question.)

### [1 point] 2. Calculate & report sparsity

Write functions to calculate the sparsity level (using the percent of buffers that are 0). Show us only the relevant code snippets. You will need to write three functions to calculate sparsity. Report the sparsity level of each parameter (weights pruned), across all pruned parameters, and for the model overall.

1. For each parameter

    ```py
    def calculate_sparsity(param):
        raise NotImplementedError()
    ```

    | Parameter | Sparsity (%) |
    | --------- | ------------ |
    | param1    |      ?       |
    | param2    |      ?       |
    | param3    |      ?       |

2. For all pruned parameters overall

    ```py
    def calculate_sparsity_overall(model, params_to_prune):
        raise NotImplementedError()
    ```

    | Sparsity (%) |
    | ------------ |
    |      ?       |

3. For the model overall
    
    ```py
    def calculate_sparsity_model(model):
        raise NotImplementedError()
    ``` 

    | Sparsity (%) |
    | ------------ |
    |      ?       |


### [1 point] 3. Calculate and report the size of the pruned model

Write a function to calculate the amount of space that a pruned model takes up when reparameterization is removed and tensors are converted to *sparse* representations. The same PyTorch pruning tutorial linked above might be helpful here. Show us only the relevant code snippet.

```py
def calculate_sparse_model_size(model):
    raise NotImplementedError()
```

**Tip:** Note that storage size actually *grows* when we first prune a model. Below we give some pointers on how to actually convert and store the parameters in a sparse format on disk.

Consider:

```py
example_model = FFN(1024, 20*20, 10, num_layers=2)
tr_loss, tr_time = train(example_model, tr_mnist, num_classes=10, log_every=500)
f = print_size_of_model(example_model,"full unpruned")
```

Now, we prune this example model by 90% all at once (we probably don't want to do this in practice, but just for illustration):

```py
example_prune_params = [
    (example_model.layers[0], 'weight'),
    (example_model.layers[1], 'weight'),
    (example_model.out, 'weight')
]

prune.global_unstructured(example_prune_params, pruning_method=prune.L1Unstructured, amount=0.9)
```

And even though the output of `calculate_sparsity` should show roughly 90% across the board, the model size will actually be larger:

```py
p = print_size_of_model(example_model, "newly pruned")
print("{0:.2f} times smaller".format(f/p))
```

Observe that model size is doubled. This is because mask buffers are stored in addition to the original parameters. So we might want to convert to sparse representations when storing on disk.


First, we remove the reparameterization, i.e. make the pruning "permanent":
```py
for p in example_prune_params:
    # p takes the form (module, 'weight')
    prune.remove(*p)
```

Now, we can easily convert the parameters to sparse representations:

```py
sd = example_model.state_dict()
for item in sd:
    # if 'weight' in item: # shortcut, this assumes you pruned (and removed reparameterization for) all weight parameters
    #     print("sparsifying", item)
    sd[item] = example_model.state_dict()[item].to_sparse()
sd
```

Now, we can check and see that when we save this sparse state dict it is indeed smaller:

```py
# now, you can save the sparse model
torch.save(sd, "model.pt")

# measure new size on disk:
print(f'{os.path.getsize("model.pt")/1e6} MB')

# notice the new size is not actually 1/10 of the size... consider why this might be the case
```

And, we can load this model and use as normal as well:
```py
# if you want to load and use this sparsified model:
sd = torch.load("model.pt") # first load state dict from disk

# now, update a new model's state dict with your stored values, converting back to dense representations as needed
new_model = FFN(1024, 20*20, 10, num_layers=2)
new_model.load_state_dict({k:(v if v.layout == torch.strided else v.to_dense()) for k,v in sd.items()})
```

Using your new disk size function for sparse representations, fill in the next row of the same table (copy the values for 0.0% sparsity from the previous table):

| Iteration | Sparsity (%) | Accuracy | Latency (s) | Disk Size (MB) |
| --------- | ------------ | -------- | ----------- | -------------- |
|     0     |      0.0%    |          |             |                |
|     1     |     33.0%    |     ?    |       ?     |         ?      |


*Note:* When calculating disk size of your sparsified models in the following sections, make sure you do *not* remove reparameterization of your "real" pruned model. Instead, make a temporary copy of your model as such (perhaps in a function):

```py
from copy import deepcopy

model_copy = deepcopy(model)
```

### [1 point] 4. Repeated unstructured magnitude pruning


Now, keep performing the same unstructured magnitude pruning of 33% of the remaining weights on the same model (*without re-training or resetting the model*). You will apply the same function as above with the same 0.33 proportion parameter.

Collect values for the rest of this table, keeping in mind that you will need to plot the results later. You might want to keep the values in Pandas DataFrames (see the section on **plotting it all together** below). Sparsity reported should be the percentage of *prunable* parameters pruned. 

| Iteration | Sparsity (%) | Accuracy | Latency (s) | Disk Size (MB) |
| --------- | ------------ | -------- | ----------- | -------------- |
|     0     |   0.0%       |          |             |                |
|     1     |   33.0%      |          |             |                |
|     2     |              |          |             |                |
|     3     |              |          |             |                |
|     4     |              |          |             |                |
|     5     |              |          |             |                |

**Tip:** Evaluating pruned models. *Assuming you have an e.g. `evaluate()` function that takes in your (pruned) model, dataloader, and possibly additional arguments*, you could use a function similar to this to evaluate models without the overhead of applying parameter masks on-the-fly (this can be useful especially if your `evaluate` function returns latency information).

```py
from copy import deepcopy

def sparse_evaluate(model, dataloader, num_classes=2):
    model_copy = deepcopy(model)
    model_copy.eval()
    
    copy_params = [(model_copy.layers[0], 'weight'),
                       (model_copy.layers[1], 'weight'),
                       (model_copy.out, 'weight')]

    # (we assume the same model architecture as the MNIST or SST-2 architecture we specify above)
    for p in copy_params:
        prune.remove(*p)
    
    return evaluate(model_copy, dataloader, num_classes)
```

**Note: this does not actually run sparse inference** - there are other frameworks that can help with that, but you're not expected to implement that for this lab.


### [1 point] 5. Iterative Magnitude Pruning (IMP) with rewinding

Now, repeat the same process as above (Exercise 2.4), but re-train the remaining weights each time (using the same hyperparameters). Importantly, you will *rewind* your model's remaining weights to their initialization in between iterations. Implementation-wise, this should look just like the above, with some extra steps (training and rewinding) between each pruning step.


Recall from class that *rewinding* refers to resetting the weights to an earlier value, rather than the most recent value during iterative magnitude pruning. Implement retraining with rewinding to the weights' values at **model initialization**, before any training or pruning was performed. (This is why we asked you to save a copy of the initialized but untrained model weights in the beginning of the lab!) You should use the same training hyperparameters each time. Collect all the same numbers as specified in the table in the previous section, putting them into a new table.

| Iteration | Sparsity (%) | Accuracy | Latency (s) | Disk Size (MB) |
| --------- | ------------ | -------- | ----------- | -------------- |
|     0     |   0.0%       |          |             |                |
|     1     |   33.0%      |          |             |                |
|     2     |              |          |             |                |
|     3     |              |          |             |                |
|     4     |              |          |             |                |
|     5     |              |          |             |                |

In your iterative magnitude pruning training loop, there are some special considerations you will need to make in order to get things working properly:

After each round of pruning, we want to *reset* all remaining weights to their values at initialization. For example, if you have a model's initial `state_dict()` saved at the relative path `"data/model_init.pt"`, you can use `torch.load("data/model_init.pt")` to reload the dict. 

Now, because the exact parameter names do not match the state dict that of the (unpruned) model at initialization, you will have to go out of your way to align them. Assuming you have e.g. `prune_param_list = ['layers.0.weight', 'layers.1.weight', 'out.weight']`, you can use:

```py
init_updated = {k + ("_orig" if k in prune_param_list else ""):v for k,v in init_weights.items()}
ffn_mnist_copy = copy.deepcopy(ffn_mnist.state_dict())
ffn_mnist_copy.update(init_updated)
ffn_mnist.load_state_dict(ffn_mnist_copy)
```

### [1 point] 7. Plotting it all together

You should report 3 plots, each of which contains two lines, one for each of the experiments you performed above: pruning without retraining, and IMP with rewinding. The three plots should have:
   - **sparsity** on the x axis and **accuracy** on the y axis. 
   - **disk space** on the x axis and **accuracy** on the y axis. 
   - **inference latency** on the x axis and **accuracy** on the y axis. 


Describe two trends depicted in your plots, and compare and contrast them. Are there trends that you expected, or didn't expect, based on discussions and lectures in class, and/or your experience? For example, is there a clear drop-off in performance at a certain sparsity level, and does that change across methods? Do latency and space on disk correspond to your expectations, and why or why not? 2-3 sentences should be sufficient.


Here is some example code you might use to plot these values using Matplotlib:

```py
import matplotlib.pyplot as plt

plt.plot(noniter_df['latency'], noniter_df['accuracy'], color="C0", label="Pruning w/o retraining")
plt.plot(IMP_df['latency'], IMP_df['accuracy'], color="C1", label="IMP w/ rewind")
plt.legend()
# YOUR CODE for titles and axis labels, etc.
plt.show()
```



## [3 points] Exercise 3: Your Model, Device, and Data


In this section, you will repeat the simple experiments from Exercise 2 on your own model, device, and data. Additionally, you will choose two of three options for practical benefits to your pruned model's accuracy and latency. You may use a different sparsity level, higher or lower than 33%, if it makes sense for your settings. Make sure to report any changes you made and why you made them. Additionally, report any challenges encountered measuring latency or storage on your device.

### [1 point] 1. Repeat Exercise 2.4 (repeated unstructured pruning) for your model, on your device and with your data.

Keep performing the same unstructured magnitude pruning of your choice of sparsity level of the remaining weights on the same model (*without re-training or resetting the model*). You will apply the same function as above with the same 0.33 proportion parameter.

Collect values for this table, keeping in mind that you will need to plot the results later. You might want to keep the values in Pandas DataFrames. Sparsity reported should be the percentage of *prunable* parameters pruned. 

| Iteration | Sparsity (%) | Accuracy | Latency (s) | Disk Size (MB) |
| --------- | ------------ | -------- | ----------- | -------------- |
|     0     |   0.0%       |          |             |                |
|     1     |      ?       |          |             |                |
|     2     |              |          |             |                |
|     3     |              |          |             |                |
|     4     |              |          |             |                |
|     5     |              |          |             |                |


### [2 points] 2. Choose two of the following three options to implement on your model, device, and data (1 point per option):

1. Implement a structured pruning technique. You may prune dimensions of matrices, attention heads, entire layers, etc. Describe your strategy and report the results in a table, adjusting the "sparsity rate" column and as needed.

    Fill in the following table with your results (choose any 2-3 pruned models to compare to the unpruned model):

    | Structure Pruned | Sparsity Rate | Accuracy | Latency (s) | Disk Size (MB) |
    | ---------------- | ------------- | -------- | ----------- | -------------- |
    | Attention heads? |               |          |             |                |
    | Layers?          |               |          |             |                |
    | Other?           |               |          |             |                |



2. Conduct a sensitivity analysis of pruning (structured or unstructured) different components of your model. For instance, what happens to your model's performance when you prune input embeddings vs hidden layer weights? Do earlier layers seem more or less important than later layers? You are not required to conduct a thorough study, but you should be able to draw a couple concrete conclusions.

    Fill in the following table with your results (choose any 2-3 pruned models to compare to the unpruned model):

    |        Pruning Technique        |  Sparsity Rate  | Accuracy | Latency (s) | Disk Size (MB) |
    | ------------------------------- | --------------- | -------- | ----------- | -------------- |
    | Unstructured, all non-embedding |  30% global     |          |             |                |
    | Structured, attention heads     |  50% per module |          |             |                |



3. Export and run your unpruned and a diverse sample of your pruned models on an inference runtime (ONNX runtime, TensorRT). Check out [the PyTorch ONNX docs](https://pytorch.org/docs/stable/onnx.html) and [this page](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html) for reference. Did you run into any challenges? Do you see latency benefits? Was anything surprising? Report inference latency and discuss.

    Fill in the following table with your results (choose any 2-3 pruned models to compare to the unpruned model):

    | Inference Runtime | Sparsity Rate | Latency (s) | Disk Size (MB) |
    | ----------------- | ------------- | ----------- | -------------- |
    | ONNX              |     0%        |             |                |
    | ONNX (pruned)     |    30%        |             |                |





## [2 points] Exercise 4: Extra Credit

### [1 point] 1. Implement an additional pruning method

Implement an additional pruning method, and report your results. For example, you might implement second-order pruning, or another structured pruning method such as [CoFi](https://github.com/princeton-nlp/cofipruning). Other potential resources: [Wanda](https://github.com/locuslab/wanda), [PyTorch semi-structured sparsity](https://pytorch.org/tutorials/prototype/semi_structured_sparse.html), [general pruning resource repository](https://github.com/he-y/awesome-pruning). Be creative! In order to get full credit, you must clearly describe your approach, and why you think it should work (but it doesn't have to work better than L1 magnitude pruning, as long as it's well-motivated). You should apply your approach in the iterative magnitude pruning paradigm, perform multiple iterations of pruning, and plot the results. Discuss how these results compare to the other methods you implemented in this lab. 3 iterations of pruning should be sufficient.


| Iteration | Sparsity (%) | Accuracy | Latency (s) | Disk Size (MB) |
| --------- | ------------ | -------- | ----------- | -------------- |
|     0     |   0.0%       |          |             |                |
|     1     |      ?       |          |             |                |
|     2     |              |          |             |                |
|     3     |              |          |             |                |


### [1 point] 2. Combining quantization and pruning
Combine quantization from Lab 2 with iterative magnitude pruning from this lab, and report your results in terms of accuracy and size on disk. You can combine the approaches however you wish, but to get full credit you must clearly describe what type of quantization you used and how exactly you combined the approaches (in what order), why you think that should work, and discuss your results. You only have to report results for one sparsity level. 


|  Pruning Technique  |    Quantization    | Sparsity Rate | Accuracy | Disk Size (MB) |
| ------------------- | ------------------ | ------------- | -------- | -------------- |
| Global unstructured | 8-bit quantization |   30% global  |          |                |


## [0.5 points] Contributions

Write down each team member's contributions to this lab by filling out the table below.

| Team Member | Contributions |
|-------------|---------------|
| Member 1    |               |
| Member 2    |               |
| Member 3    |               |
| Member 4    |               |

## Grading and submission (10 points + 2 extra credit)
Submit your answers to all the above questions to Canvas as a write-up in pdf format. This assignment is worth 10 points (or 10% of your final grade for the class) with a total possible points of 12 with extra credit. You may use this file as a starting point.
