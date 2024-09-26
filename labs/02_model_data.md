Lab 2: Benchmarking Model, Data, Hardware & Energy
===

In this lab you will extend the same benchmarking analysis from Lab 1 to a model, dataset, and (optionally) hardware platform that you will be using for your class project. You will also measure the inference energy of the model on different hardware platforms, and convert those measurements into estimated carbon emissions.

> Note: While you are not required to have your model deployed on the hardware you plan to use for your final project at this stage, you are strongly encouraged to do it (there is extra credit available) so that you don't have to work during Fall Break (and we don't want you to)!

We do not expect that you will be re-implementing the model from scratch. Depending on your project, you may be working with an existing model that you plan to optimize or fine-tune.


The main goals of this exercise are: 
 1. Proficiency training and evaluating more advanced models and task setups with respect to efficiency and accuracy.
 2. (In some cases) Experience moving to a second hardware platform.
 3. Ironing out kinks in your project proposal & measurable progress towards your final project.
 4. Measuring the inference energy and converting those measurements into estimated carbon emissions.


> **Note**: This is a group lab assignment; you will submit one report per project group. 
<br><br> 
We require that you submit your report as a pdf. We **do not** want you to submit _all_ of your code (leaving us with 40 pages of code and outputs and delaying your grading). Please only show us necessary code snippets where explicitly requested. You are free to use this file as a starting point.

Preliminaries
----

List your project team name & members.

| Team Name | [Your Team Name] |
|-----------|------------------|
| Member 1  | [Your Name]      |
| Member 2  | [Your Name]      |
| Member 3  | [Your Name]      |
| Member 4  | [Your Name]      |


Base Model and Evaluation [1 point]
----

1. [0.5 points] Clearly describe the task, dataset and split, and model that you will be using, including specific references to the relevant papers and code.

1. [0.5 points] Compute the accuracy for your task. What accuracy do you obtain on your task, and using what hyperparameters? This will be your "base model". 

    > Note that the most appropriate evaluation for your task might not be accuracy; please describe how end-task performance is measured for your dataset. "Accuracy" in the rest of the report will refer to this metric.



Benchmarking Base Model [4 points]
----

3. [1.5 points] Now, implement three methods for benchmarking the efficiency of this model. Explain your approach. If your approach for measuring inference latency is different for your hardware, explain how you measured inference latency on your device. If you are doing it using code, show us only the relevant code snippets for each method.

    - **Inference Latency:** Measure the average time it takes to classify each example. You should run this a few times, and throw away the first measurement to allow for initial warmup (e.g. caching, etc.) 

        ```python
        def measure_inference_latency(model, data):
            # your code here
            return latency
        ```

    - **Parameter count:** Write a function to compute the number of parameters in your model. You should be able to use the same function as you implemented in Lab 1.

        ```python
        def count_parameters(model):
            # your code here
            return num_parameters
        ```

    - **FLOPs:** Write a function to compute the number of floating-point operations that need to be performed in order to do inference on one example in your model. This time, feel free to use resources you find on the web, but be sure to cite them in your writeup. Clearly explain the approach you took and explain any decision points.

        ```python
        def count_flops(model, data):
            # your code here
            return flops
        ```

4. [0.5 points] What is the exact hardware that you are using to benchmark your code? Report the CPU and RAM, in as much detail as is available from your operating system.

5. [1 point] Report the average inference latency of your model and plot the first ten iterations of inference latency (including the warm-up ones). Is the variance high or low? Did you notice any outliers? Was the first iteration of inference slower than the others? Try to explain any phenomena you note.

6. [0.5 points] Report the parameter count of your model.

7. [0.5 points] Report the number of FLOPs that your model requires to perform inference. Does this align with your expectations? Why or why not?

Varying the input size [2 points]
----

8. [1.5 points] Now, try varying the input size (input resolution) of your base model. Downsample using two different methods, across 2 different resolutions. Report the parameter counts, FLOPs, and inference latency, and generate three plots (each plot showing both methods of downsizing) for each of the following:

    - FLOPs on the x axis and accuracy on the y axis
    - Latency on the x axis and accuracy on the y axis
    - FLOPs on the x axis and latency on the y axis

9. [0.5 point] Explain why you chose each method of downsizing. How do the two different transformations compare in terms of accuracy and efficiency?

Varying number of parameters [1.5 points]
----

Now, try varying the number of parameters in your base model, or using a larger/smaller model of the same family. If you are modifying your base model (removing layers, etc.), try to come up with a method that will have the smallest negative impact on accuracy. The specific way that you do this will depend on the model. 

You don't need to do anything fancy, you just need to explain what you did and why (note that your accuracy may fall drastically).

> If you are planning on training or fine-tuning models for your project, feel free to re-train or fine-tune with smaller parameter sizes; otherwise, you don't need to train here.

10. [0.5 points] Explain how you decided to remove paramters, and justify your decisions. Why did you choose this method over other methods?

11. [1 point] Train and evaluate your model using at least two paramter sizes. Report the same three plots as in (8) above. Discuss your results.

Measuring Energy Use [1 point]
----

At this stage of the project, we do not expect you to measure your energy utilization on the hardware you plan to use for your final project. For this lab, you are expected to measure energy using [CodeCarbon](https://github.com/mlco2/codecarbon).

   
12. [0.5 points] For your base model, any one model with varied input size, and any one model with a varied number of parameters, measure the energy use of a single inference using CodeCarbon. You can measure this by running multiple inferences and averaging the results. You are free to run this on your local machine or your project hardware. Please specify the hardware you are using.


Report the following:

**Hardware platform:** __________

| Model | Input size | Number of parameters | Energy (kWh) | Carbon emissions (kg) |
|-------|------------|----------------------|--------------|-----------------------|
| Base  |            |                      |              |                       |
| Smaller input  |            |                      |              |                       |
|  Fewer parameters     |            |                      |              |                       |


13. [0.5 points] Pounds or grams of carbon are not very intuitive measures. It can be useful to find points of comparison, emissions due to other common activities to compare to. Do some research to find some emissions of other activities to provide a point of reference. Are you suprised by the results? Why or why not?


Extra Credit: Deploying on Hardware [2 points]
----
14. If you benchmarked everything on the hardware you plan to use for your final project, you will receive 2 points of extra credit. Please provide a brief description how you deployed your model(s) on device, and any challenges you faced.


Contributions [0.5 points]
----

15. [0.5 points] Write down each team member's contributions to this lab by filling out the table below.

| Team Member | Contributions |
|-------------|---------------|
| Member 1    |               |
| Member 2    |               |
| Member 3    |               |
| Member 4    |               |


Grading and submission (10 points + 2 points extra credit)
----
Submit your answers to all the above questions to Canvas as a write-up in pdf format. This assignment is worth 12 total possible points, with 10 points assigned for the main questions and 2 points assigned for the extra credit question.


