Lab 2: Benchmarking Your Model, Data, and Hardware
===
In this lab you will extend the same analysis from Lab 1 to a model, dataset, and hardware platform that you will be using for your class project. 
In many cases you may be taking existing model code and working with that (i.e. depending on your project, we do not expect that you will be re-implementing the model from scratch.)
The main goals of this exercise are: 
 1. Proficiency training and evaluating more advanced models and task setups with respect to efficiency and accuracy; 
 2. (In many cases) experience moving to a second hardware platform.
 3. Ironing out kinks in your project proposal & measurable progress towards your final project.

*Note*: This is a group lab assignment; you will submit one report per project group.

Preliminaries
----
0. List your project team name & members. Clearly describe the task, dataset and split, and model that you will be using, including specific references to the relevant papers and code.


Model and Evaluation
----
Compute classification accuracy for your task.
1. Note that the most appropriate evaluation for your task might not be accuracy; please describe how end-task performance is measured for your dataset. "Accuracy" in the rest of the report will refer to this metric.
2. What accuracy do you obtain on your task, and using what hyperparameters? This will be your "base model." 

Now, implement three methods for benchmarking the efficiency of this model:
- **Latency:** Measure inference latency. Measure the average time it takes to classify each example. You should run this a few times, and throw away the first measurement to allow for initial warmup (e.g. caching, etc.) 
- **Parameter count:** Write a function to compute the number of parameters in your model. You should be able to use the same function as you implemented in Lab 1.
- **FLOPs:** Write a function to compute the number of floating-point operations that need to be performed in order to do inference on one example in your model. This time, feel free to use resources you find on the web, but be sure to cite them in your writeup. Clearly explain the approach you took and explain any decision points.

3. What is the exact hardware that you are using to benchmark your code? Report the CPU and RAM, in as much detail as is available from your operating system.
4. Report the average inference latency of your model. Is the variance high or low? Did you notice any outliers? Was the first iteration of inference slower than the others? Try to explain any phenomena you note.
5. Report the parameter count of your model.  
6. Report the number of FLOPs that your model requires to perform inference. Does this align with your expectations? Why or why not?

Varying the input size
----

7. Now, try varying the input size (input resolution) of your base model. Downsample using at least two different methods, across a variety of resolutions. Report the parameter counts, FLOPs, and inference latency, and generate three plots:
    - FLOPs on the x axis and accuracy on the y axis
    - Latency on the x axis and accuracy on the y axis
    - FLOPs on the x axis and latency on the y axis
8. Explain why you chose each method of downsizing. How do the two different transformations compare in terms of accuracy and efficiency?

Varying number of parameters
----
Now, try varying the number of parameters in your base model. You don't need to do anything fancy, you just need to explain what you did and why (note that your accuracy may fall drastically). 
Try to come up with a method that will have the smallest negative impact on accuracy. The specific way that you do this will depend on the model. If you are planning on training or fine-tuning models for your project, 
feel free to re-train or fine-tune with smaller parameter sizes; otherwise, you don't need to train here.

9. Explain how you decided to remove paramters, and justify your decisions. Why did you choose this method over other methods?
10. Train and evaluate your model using a variety of paramter sizes. Report the same three plots as in (7) above. Discuss your results.
   
Putting it all together
----
Now, experiment with different settings combining different input sizes with different parameter counts. 

11. Generate the same three plots as above, reporting the results of your experiments. Do you notice any interesting trends? What combination of input resolution and parameter count seems to correspond to the best trade-off between accuracy and efficiency? Do you notice differences between latency and FLOPs? Explain your experimental process and discuss your results.

Grading and submission (10 points)
----
Submit your answers to all the above questions to Canvas as a write-up in pdf format. This assignment is worth 10 points, graded as follows: 
- **Submission [2 points]:** Assignment is submitted on time.
- **Basic requirements [5 points]:** Answers to all the questions (including all requested plots) are included in the write-up.
- **Report [2 points]:** Report is well-written and organized with minimal typos and formatting mistakes, and answers to all requested questions are easy to find. Plots are readable and well-labeled. Your results should be easily reproducible from the details included in your report.
- **Thoughtful discussion [1 point]:** The discussion of results in your write-up is thoughtful: it indicates meaningful engagement with the exercise and it is clear that you put effort into thinking through and explaining your results.
- **Contributions [required]:** List out the specific aspects of the report that each member contributed to.
