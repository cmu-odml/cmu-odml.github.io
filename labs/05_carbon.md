Lab 4: Carbon
===
In this lab you will gain experience measuring the inference energy use of ML models on different hardware platforms, and converting those measurements into estimated carbon emissions. **You will do this lab as a group project with your class project team**, and you only need to submit a single report jointly for your team.

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

[describe]

If you are able to measure energy use using both methods for both platforms, do so and compare your findings. You may not be able to use both methods for both hardware platforms, either due to not having access to the power source of the hardware (in the case of a server), or due to the hardware platform not having on-board support for power measurement (as is the case for the Jetson Nano 2GB). This is ok, but please clearly state why this is the case for each platform, and perform energy measurement using at least one method for each platform.

Converting energy to estimated carbon emissions
---


Discussion
---


Extra Credit
---
#### 1. Estimate embodied emissions for your models [1 point]
Do some research to investigate the embodied carbon emissions for both hardware platforms you chose, and report your findings. You don't need to incorporate these numbers into your inference-level estimate of emissions, you can just report the overall estimated emissions required to produce and recycle the hardware. Cite your sources.

#### 2. Use WattTime to get a more precise estimate of carbon emissions [2 points]


Grading and submission (10 points + 3 extra credit)
----
Submit your answers to all the above questions to Canvas as a write-up in pdf format. This assignment is worth 10 points 
(or 10% of your final grade for the class), distributed as follows: 
- **Submission [2 points]:** Assignment is submitted on time.
- **Basic requirements [5 points]:** Answers to all the questions (including all requested tables) are included in the write-up. 
- **Report [2 points]:** Report is well-written and organized with minimal typos and formatting mistakes, and answers to all requested questions are easy to find. Tables are readable and well-labeled. Your results should be easily reproducible from the details included in your report.
- **Thoughtful discussion [1 points]:** The discussion of results in your write-up is thoughtful: it indicates meaningful engagement with the exercise and it is clear that you put effort into thinking through and explaining your results.
- **Extra Credit [3 points]:** See above for description of possible extra credit.
