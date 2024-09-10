Lab 2: Benchmarking Model, Data & Energy
===
In this lab you will extend the same benchmarking analysis from Lab 1 to a model, dataset, and hardware platform that you will be using for your class project. You will also measure the inference energy of the model on different hardware platforms, and convert those measurements into estimated carbon emissions.


We do not expect that you will be re-implementing the model from scratch. Depending on your project, you may be working with an existing model that you plan to optimize or fine-tune.


The main goals of this exercise are: 
 1. Proficiency training and evaluating more advanced models and task setups with respect to efficiency and accuracy.
 2. (In some cases) Experience moving to a second hardware platform.
 3. Ironing out kinks in your project proposal & measurable progress towards your final project.
 4. Measuring the inference energy and converting those measurements into estimated carbon emissions.


> **Note**: This is a group lab assignment; you will submit one report per project group. 
<br><br> 
We recommend that you submit your report as a pdf from a Jupyter notebook. We **do not** want you to submit _all_ of your code (leaving us with 40 pages of code and outputs and delaying your grading), but rather use the notebook as a markdown text editor, only showing us code snippets where necessary. You are free to use this file as a starting point.

Preliminaries
----

List your project team name & members.

| Team Name | [Your Team Name] |
|-----------|------------------|
| Member 1  | [Your Name]      |
| Member 2  | [Your Name]      |
| Member 3  | [Your Name]      |
| Member 4  | [Your Name]      |


Base Model and Evaluation [2 points]
----

1. [1 point] Clearly describe the task, dataset and split, and model that you will be using, including specific references to the relevant papers and code.

1. [1 point] Compute the accuracy for your task. What accuracy do you obtain on your task, and using what hyperparameters? This will be your "base model". 

    > Note that the most appropriate evaluation for your task might not be accuracy; please describe how end-task performance is measured for your dataset. "Accuracy" in the rest of the report will refer to this metric.



Benchmarking Base Model [8 points]
----

3. [3 points] Now, implement three methods for benchmarking the efficiency of this model. Show us only the relevant code snippets, and explain your approach.
    - **Inference Latency:** Measure the average time it takes to classify each example. You should run this a few times, and throw away the first measurement to allow for initial warmup (e.g. caching, etc.) 
    - **Parameter count:** Write a function to compute the number of parameters in your model. You should be able to use the same function as you implemented in Lab 1.
    - **FLOPs:** Write a function to compute the number of floating-point operations that need to be performed in order to do inference on one example in your model. This time, feel free to use resources you find on the web, but be sure to cite them in your writeup. Clearly explain the approach you took and explain any decision points.

4. [1 point] What is the exact hardware that you are using to benchmark your code? Report the CPU and RAM, in as much detail as is available from your operating system.

5. [2 points] Report the average inference latency of your model and plot the first ten iterations of inference latency (including the warm-up ones). Is the variance high or low? Did you notice any outliers? Was the first iteration of inference slower than the others? Try to explain any phenomena you note.

6. [1 point] Report the parameter count of your model.

7. [1 point] Report the number of FLOPs that your model requires to perform inference. Does this align with your expectations? Why or why not?

Varying the input size [4 points]
----

8. [3 points] Now, try varying the input size (input resolution) of your base model. Downsample using two different methods, across 3 different resolutions. Report the parameter counts, FLOPs, and inference latency, and generate three plots (each plot showing both methods of downsizing) for each of the following:

    - FLOPs on the x axis and accuracy on the y axis
    - Latency on the x axis and accuracy on the y axis
    - FLOPs on the x axis and latency on the y axis

9. [1 point] Explain why you chose each method of downsizing. How do the two different transformations compare in terms of accuracy and efficiency?

Varying number of parameters [3 points]
----

Now, try varying the number of parameters in your base model, or using a larger/smaller model of the same family. If you are modifying your base model (removing layers, etc.), try to come up with a method that will have the smallest negative impact on accuracy. The specific way that you do this will depend on the model. 

You don't need to do anything fancy, you just need to explain what you did and why (note that your accuracy may fall drastically).

> If you are planning on training or fine-tuning models for your project, feel free to re-train or fine-tune with smaller parameter sizes; otherwise, you don't need to train here.

10. [1 point] Explain how you decided to remove paramters, and justify your decisions. Why did you choose this method over other methods?

11. [2 points] Train and evaluate your model using at least two paramter sizes. Report the same three plots as in (8) above. Discuss your results.

Measuring Energy Use [2 points]
----

At this stage of the project, we do not expect you to have deployed your models onto the hardware you plan to use for your final project. For this lab, you are expected to measure energy using [CodeCarbon](https://github.com/mlco2/codecarbon). 
   
12. [1 point] For your base model, any one model with varied input size, and any one model with a varied number of parameters, measure the energy use of a single inference using CodeCarbon. You can measure this by running multiple inferences and averaging the results.


Report the following:

| Model | Input size | Number of parameters | Energy (kWh) | Carbon emissions (kg) |
|-------|------------|----------------------|--------------|-----------------------|
| Base  |            |                      |              |                       |
| Smaller input  |            |                      |              |                       |
|  Fewer parameters     |            |                      |              |                       |


13. [1 point] Pounds or grams of carbon are not very intuitive measures. It can be useful to find points of comparison, emissions due to other common activities to compare to. Do some research to find some emissions of other activities to provide a point of reference. Are you suprised by the results? Why or why not?


Contributions [1 point]
----

14. [1 point] Write down each team member's contributions to this lab by filling out the table below.

| Team Member | Contributions |
|-------------|---------------|
| Member 1    |               |
| Member 2    |               |
| Member 3    |               |
| Member 4    |               |


Grading and submission (20 points)
----
Submit your answers to all the above questions to Canvas as a write-up in pdf format. This assignment is worth 20 points. 


