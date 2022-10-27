Lab 3: Pruning
===
This lab is an opportunity to explore different pruning strategies and settings, with the goal of becoming familiar with unstructured, iterative and non-iterative pruning in PyTorch, and some of the trade-offs of using such an approach for model compression.

Throughout this lab, we will use the term **sparsity level** to refer to the percent of the original model's *prunable* weights (**weight parameters, excluding bias parameters**) that have been pruned.

If you are compute-resource-constrained (i.e. your personal laptop takes a really long time to perform a training run with the base model), you can change the hyperparameters somewhat to reduce computational burden -- e.g. training epochs, hidden size, but please clearly report what you did, try to keep the changes minimal, and be consistent throughout the assignment.

Preliminaries & Setup
---
0. Share the hardware specs for the machine you will be using to run the experiments in this lab.
1. Copy over relevant code for training MNIST from Lab 2 (just the "Lab 2" model), including code for evaluation (in particular, accuracy, latency, and size on disk), *but don't train yet*!

| hyperparameter  | value |
| --------------- | ----- |
| learning rate   | 0.001 |
| batch size      | 64    |
| hidden size     | 1024  | 
| # hidden layers | 2     |
| training epochs | 2     |

Recall from Lab 2 that you can measure model size on disk in this way:

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
f=print_size_of_model(model1, "model1")
q=print_size_of_model(model2, "model2")
print("{0:.2f} times smaller".format(f/q))
```

2. Initialize your model. Before training, **SAVE** your model's initial (random) weights. You will use them later for iterative pruning
3. Now train the base model and report:
   - dev accuracy, 
   - inference latency,
   - number of parameters,
   - space needed for storage (in MB) of `state_dict()` on disk

| Iteration | Sparsity (%) | Accuracy | Latency (s) | Disk Size (MB) |
| ------- | ------- | ------ | ------- | ------ |
|     0   |   0.0%  |    ?    |    ?     |    ?    |

Also take a look at your model's `named_parameters()`. You'll need these later (no need to put in the table).


Unstructured magnitude pruning
---
First, you will perform global, unstructured magnitude (L1) pruning on the model to a sparsity level of **33%**. Prune just the weight parameters (not biases). 
You should be able to use the `global_unstructured` pruning method in the PyTorch prune module.
For usage examples, see the [PyTorch pruning tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html). 

Example input:
```py
[(model.layers[0], 'weight'),
(model.layers[1], 'weight'),
(model.out, 'weight')]
```
or
```py
[(m[1], "weight") for m in model.named_modules() if len(list(m[1].children()))==0]
```
Take a look at your `model.named_parameters()` again, and your `model.named_buffers()`. (Just observe, no need to answer a question.)

4. Write functions to calculate the sparsity level (using the percent of buffers that are 0):
    -  for each layer,
    -  for all pruned layers, and
    -  for the model overall
   **And report each of these values:** the sparsity level at each layer, across all pruned layers, and for the model overall.
5. Write a function to calculate the amount of space that a pruned model takes up when reparameterization is removed and tensors are converted to *sparse* representations.

**Tip:** Note that storage size actually *grows* when we first prune a model. Below we give some pointers on how to actually convert and store the parameters in a sparse format on disk.

Consider:
```py
example_model = FFN(1024, 20*20, 10, num_layers=2)
tr_loss, tr_time = train(example_model, tr_mnist, num_classes=10, log_every=500)
f=print_size_of_model(example_model,"full unpruned")
```
Now, we prune this example model by 90% all at once (we probably don't want to do this in practice):
```py
example_prune_params = [(example_model.layers[0], 'weight'),
                           (example_model.layers[1], 'weight'),
                           (example_model.out, 'weight')]

prune.global_unstructured(example_prune_params, pruning_method=prune.L1Unstructured, amount=0.9)
```
And even though the output of `calculate_sparsity` should show roughly 90% across the board...
```py
p=print_size_of_model(example_model, "newly pruned")
print("{0:.2f} times smaller".format(f/p))
```
Observe that model size is doubled.
This is because mask buffers are stored in addition to the original parameters.
So we might want to convert to sparse representations when storing on disk.
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

# notice not actually 1/10 of the size but oh well
```

And, we can load this model and use as normal as well:
```py
# if you want to load and use this sparsified model:
sd = torch.load("model.pt") # first load state dict from disk

# now, update a new model's state dict with your stored values, converting back to dense representations as needed
new_model = FFN(1024, 20*20, 10, num_layers=2)
new_model.load_state_dict({k:(v if v.layout == torch.strided else v.to_dense()) for k,v in sd.items()})
```

Using your new disk size function, fill in the next row of the same table:

| Iteration | Sparsity (%) | Accuracy | Latency (s) | Disk Size (MB) |
| ------- | ------- | ------ | ------- | ------ |
|     0   |   0.0%  |        |         |        |
|     1   |   33.0%   |    ?    |    ?    |    ?    |


Repeated unstructured magnitude pruning
---
Now, keep performing the same unstructured magnitude pruning of 33% of the remaining weights on the same model (without re-training or resetting the model). 
You will apply the same function as above with the same 0.33 proportion parameter.

6. Collect values for the rest of this table, keeping in mind that you will need to plot the results later. You might want to keep the values in Pandas DataFrames (see the section on **plotting it all together** below.) Sparsity reported should be the percentage of *prunable* parameters pruned. 

| Iteration | Sparsity (%) | Accuracy | Latency (s) | Disk Size (MB) |
| ------- | ------- | ------ | ------- | ------ |
|     0   |   0.0%  |        |         |        |
|     1   |   33.0% |        |         |        |
|     2   |         |        |         |        |
|     3   |         |        |         |        |
|     4   |         |        |         |        |
|     5   |         |        |         |        |
|     6   |         |        |         |        |
|     7   |         |        |         |        |
|     8   |         |        |         |        |
|    10   |         |        |         |        |
|    11   |         |        |         |        |
|    12   |         |        |         |        |
|    13   |         |        |         |        |
|    14   |         |        |         |        |
|    15   |         |        |         |        |
|    16   |         |        |         |        |
|    17   |         |        |         |        |
|    18   |         |        |         |        |
|    19   |         |        |         |        |
|    20   |         |        |         |        |

**Tip:** Evaluating pruned models. *Assuming you have an e.g. `evaluate()` function that takes in your (pruned) model, dataloader, and possibly additional arguments*, you could use a function similar to this to evaluate models without the overhead of applying parameter masks on-the-fly (this can be useful especially if your `evaluate` function returns latency information).
```py
from copy import deepcopy

def sparse_evaluate(model, dataloader, num_classes=2):
    model_copy = deepcopy(model)
    model_copy.eval()
    
    copy_params = [(model_copy.layers[0], 'weight'),
                       (model_copy.layers[1], 'weight'),
                       (model_copy.out, 'weight')]
    # (we assume the same model architecture as the MNIST architecture we specify above)
    for p in copy_params:
        prune.remove(*p)
    
    return evaluate(model_copy, dataloader, num_classes)
```
**Note: this does not actually run sparse inference** - there are other frameworks that can help with that, but you're not expected to implement that for this lab.

Iterative magnitude pruning (IMP)
---
Now, repeat the same process as above, but re-train the remaining weights each time (using the same hyperparameters). 
You will experiment with two settings: Re-training without rewinding, and re-training with rewinding. Implementation-wise,
this should look just like the above, with some extra steps (training, and optionally rewinding) between each pruning step.

7. **IMP without rewinding:** Continue training the unpruned weights starting from their current value at each iteration, the value that was used to determine which weights to prune from the last iteration. Collect all the same numbers as specified in the table in the previous section. You should use the same training hyperparameters.

8. **IMP with rewinding:** Recall from class that *rewinding* refers to resetting the weights to an earlier value, rather than the most recent value during iterative magnitude pruning. Implement retraining with rewinding to the weights' values at **model initialization**, before any training or pruning was performed. (This is why we asked you to save a copy of the initialized but untrained model weights in the beginning of the lab!) You should use the same training hyperparameters. Collect all the same numbers as specified in the table in the previous section.

In your iterative magnitude pruning training loop, there are some special considerations you will need to make in order to get things working properly:

Recall that, after each round of pruning, we want to *reset* all remaining weights to their values at initialization. For example, if you have a model's `state_dict()` saved at the relative path `"data/model_init.pt"`, you can use `torch.load("data/model_init.pt")` to reload the dict. 

Now, because the exact parameter names do not match the state dict that of the (unpruned) model at initialization, you will have to go out of your way to align them. Assuming you have e.g. `prune_param_list = ['layers.0.weight', 'layers.1.weight', 'out.weight']`, you can use:
```py
init_updated = {k + ("_orig" if k in prune_param_list else ""):v for k,v in init_weights.items()}
ffn_mnist_copy = copy.deepcopy(ffn_mnist.state_dict())
ffn_mnist_copy.update(init_updated)
ffn_mnist.load_state_dict(ffn_mnist_copy)
```

Plotting it all together
---
You should report 3 plots, each of which contains a line corresponding to each of the experiments you performed above: pruning without retraining, IMP without rewinding, and IMP with rewinding. The three plots should have:
   - **accuracy** on the x axis and **sparsity** on the y axis. 
   - **accuracy** on the x axis and **disk space** on the y axis. 
   - **accuracy** on the x axis and **inference latency** on the y axis. 

**Tip:** If you have three Pandas DataFrames each containing columns for: 1) iteration number, 2) sparsity (of prunable parameters), 3) accuracy, 4) inference latency, and 5) size on disk, you can plot e.g. accuracy vs latency using a more elaborate version of this code (i.e. with a title and axis labels):
Here is some example code you might use to plot these values using Matplotlib:
```py
import matplotlib.pyplot as plt

plt.plot(noniter_df['iteration'], noniter_df['latency'], color="C0", label="Pruning w/o retraining")
plt.plot(IMP_df['iteration'], IMP_df['latency'], color="C1", label="IMP w/ rewind")
plt.plot(stdprune_df['iteration'], stdprune_df['latency'], color="C2", label="IMP no rewind")
plt.legend()
# YOUR CODE for titles and axis labels, etc.
plt.show()
```

Discussion
---
- Choose two of the plots, describe the trends they depict, and compare and contrast the plots. Are there trends that you expected, or didn't expect, based on discussions and lectures in class, and/or your experience? For example, is there a clear drop-off in performance at a certain sparsity level, and does that change across methods? Do latency and space on disk correspond to your expectations, why or why not?
- In 4-5 sentences, pose one small follow-up experiment that you might run, based on these initial results. Your motivation (based on these results), hypothesis and methodology for testing that hypothesis should be clear. You do not need to run the experiment. (This can overlap with extra credit, if you choose to implement extra credit.)

Extra Credit
---
#### 1. Implement a custom pruning method [1 point]
Implement an additional pruning method, and report your results. For example, you might implement structured pruning, second-order pruning, or anything else. Be creative! In order to get full credit, you must clearly describe your approach, and why you think it should work (but it doesn't have to work better than L1 magnitude pruning, as long as it's well-motivated). You should apply your approach in the iterative magnitude pruning paradigm, perform multiple iterations of pruning, and plot the results. Discuss how these results compare to the other methods you implemented in this lab.

#### 2. Combining quantization and pruning [1 point]
Combine quantization from Lab 2 with iterative magnitude pruning from this lab, and report your results in terms of accuracy and size on disk. You can combine the approaches however you wish, but to get full credit you must clearly describe what type of quantization you used and how exactly you combined the approaches, why you think that should work, and discuss your results. You only have to report results for one sparsity level. 


Grading and submission (10 points + 2 extra credit)
----
Submit your answers to all the above questions to Canvas as a write-up in pdf format. This assignment is worth 10 points 
(or 10% of your final grade for the class), distributed as follows: 
- **Submission [2 points]:** Assignment is submitted on time.
- **Basic requirements [5 points]:** Answers to all the questions (including all requested tables) are included in the write-up. 
- **Report [2 points]:** Report is well-written and organized with minimal typos and formatting mistakes, and answers to all requested questions are easy to find. Tables are readable and well-labeled. Your results should be easily reproducible from the details included in your report.
- **Thoughtful discussion [1 points]:** The discussion of results in your write-up is thoughtful: it indicates meaningful engagement with the exercise and it is clear that you put effort into thinking through and explaining your results.
- **Extra Credit [2 points]:** See above for description of possible extra credit.
