Lab 4: Energy & Carbon
===
In this lab you will gain experience measuring the inference energy use of ML models on different hardware platforms, and converting those measurements into estimated carbon emissions.
-  **You will do this lab as a group project with your class project team**, and you only need to submit a single report jointly for your team.
- You need to collect a [USB Multimeter](https://www.amazon.com/gp/product/B07DCTG6LH/) or [Kill-a-Watt](https://www.amazon.com/P3-P4400-Electricity-Usage-Monitor/dp/B00009MDBU/) from us to complete this lab.
- This is an _**inference**_ only lab, but there are lots of variables that affect power draw, so you will need to run inference multiple times.
- You now should be familiar with lots of the issues, so this lab is a bit more open ended.

Preliminaries
---
In this lab you will benchmark energy use and estimate carbon emissions for two models on two hardware platforms. 
1. First, choose **two models that you expect to have different energy requirements**. For example, you might choose to use one of the feed-forward networks from the previous lab assignments, and one of the models you're using for your class project. 
 - Clearly describe each model, including details that would be relevant to its expected energy use such as the underlying architecture, number of parameters, etc.
2. Now, specify **two hardware platforms** upon which you will evaluate the energy use of both models. For example, you might use your team's Raspberry Pi or NVIDIA Jetson as one platform, and a team member's personal laptop or server as the second platform.
- Clearly describe the hardware specs for the two platforms you will be using to run the experiments in this lab.
3. Which model do you expect to require more energy to perform an inference on each hardware platform, and why? Consider aspects such as model architecture, input resolution and format, and capabilities of the underlying hardware.

Measuring energy use
---
There are two ways to measure energy use: using a power meter at the wall, and (for some hardware) using on-board support for power monitoring. 

**Power meter**
- *USB Multimeter* Your multimeter fits directly into the USB plug.  It measures both instantaneous values (Amp, Volt, Watt) and over time (Watt Hours).  You can also reset the accumulator.
- *Kill-a-Watt* requires longer runtimes to get a KWH reading
This is a manual process so there will be some error in measurement, but repeated and longer experiments help reduce noise

**On-board / command line**
The exact tools for this will vary depending on the hardware platform. On most platforms, you can use [PowerTOP](https://github.com/fenrus75/powertop). For Intel-based platforms, you can use the [RAPL power meter](https://web.eece.maine.edu/~vweaver/projects/rapl/). On NVIDIA platforms, you can use `nvidia-smi`.

You may not be able to use both methods for both hardware platforms, either due to not having access to the power source of the hardware (in the case of a server), or due to the hardware platform not having on-board support for power measurement (as is the case for the Jetson Nano 2GB). This is ok, but please clearly state why this is the case for each platform, and perform energy measurement using at least one method for each platform.

Multiply energy draw by inference time to get an estimate of energy required per inference (you can average over input size).

4. **Benchmark the first model**
  * How many times do you need to run inference to get a stable estimate?
  * What batch size did you choose, and why?
  * When did you start/stop measuring power, and why?
5. **Benchmark the second model**
  * Did this model perform as you expected, decribed in (3)? If not, why do you think this is?
6. **Make a change to your environment** (e.g. OS, background processes, etc) that you think will affect power draw. If you are able to measure energy use using both methods (wall and command line), you might consider trying this. Compare and discuss your findings. 
  * What change did you make? Why did you think it would help?
  * How much did it change your results from (5)?

**Example Table:**
| Model Details | Hardware | Measurement method |Batch | Watt-Hours/Mins | Std Dev |
| ------------- | ----- | ----| - | --------------- | ------- |
| Model 1       | HW1      | Multimeter   |    |                |         |
| Model 2       | HW1      |     |   |                |         |
| Model 1       | HW2      |      |  |                |         |
| Model 2       | HW2      |     |   |                |         |
| Changes       |          |      |   |               |         |

Column one requires corresponding justification and additional details (above).

Converting energy to estimated carbon emissions
---
Now, convert watt-hours to estimated emissions by multiplying by carbon intensity. You can use [this resource](https://www.epa.gov/egrid/power-profiler#/) to get a coarse-grained estimate of energy intensity for Pittsburgh.

7. Add this estimate as a column to your table.
8. Pounds or grams of carbon are not very intuitive measures. It can be useful to find points of comparison, emissions due to other common activities to compare to. Do some research to find some emissions of other activities to provide a point of reference. Are you suprised by the results? Why or why not?

Discussion
---
Answer all bullets above with justifications and explanations of counter-intuitive results.

Extra Credit
---
#### 1. Estimate embodied emissions for your models [1 point]
Do some research to investigate the embodied carbon emissions for both hardware platforms you chose, and report your findings. You don't need to incorporate these numbers into your inference-level estimate of emissions, you can just report the overall estimated emissions required to produce and recycle the hardware. Cite your sources.

#### 2. Use WattTime to get a more precise estimate of carbon emissions [2 points]
The [WattTime API](https://www.watttime.org/api-documentation/#introduction) provides more fine-grained information on carbon intensity for a given location (e.g. latitude/longitude) and time. Use calls to this API to get a better estimate of carbon intensity based on the time and location where your experiments were run, and add a column to the table. Try varying the time and location. Do the emissions change? Hypothesize why or why not, based on the times and locations you tried.


Grading and submission (10 points + 3 extra credit)
----
Submit your answers to all the above questions to Canvas as a write-up in pdf format. This assignment is worth 10 points 
(or 10% of your final grade for the class), distributed as follows: 
- **Submission [2 points]:** Assignment is submitted on time.
- **Basic requirements [5 points]:** Answers to all the questions (including all requested tables) are included in the write-up. 
- **Report [2 points]:** Report is well-written and organized with minimal typos and formatting mistakes, and answers to all requested questions are easy to find. Tables are readable and well-labeled. Your results should be easily reproducible from the details included in your report.
- **Thoughtful discussion [1 points]:** The discussion of results in your write-up is thoughtful: it indicates meaningful engagement with the exercise and it is clear that you put effort into thinking through and explaining your results.
- **Extra Credit [3 points]:** See above for description of possible extra credit.
