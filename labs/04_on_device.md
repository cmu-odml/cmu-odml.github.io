Lab 4: On-Device
===
In this lab you will gain experience setting up and running models on a new, small device: Either the Jetson Nano or Raspberry Pi, depending on the device that has been assigned to your group. You will re-run code and experiments for quantization and pruning that you developed in Lab 2 and Lab 3, and compare your experiences and results to the original hardware (laptop or server) that you used for those assignments. You will also practice researching more detailed hardware specifications in order to understand what should or should not work on a given device and why.

Preliminaries & Setup
---
Before coming to class, you should already have flashed your device with the latest operating system and everyone in your group
should be able to ssh in to the device.

### Flashing the device
You can use [BalenaEtcher](https://www.balena.io/etcher/) to flash the SD card.

#### Raspberry Pi 4 (8GB)
- We recommend flashing the latest version of Ubuntu Desktop for Raspberry Pi, which you can find [here](https://ubuntu.com/download/raspberry-pi).
- You can save some memory by using Ubuntu Server (no graphical interface), but note that WiFi does not work out-of-the-box on Ubuntu Server so you will need a hard-wired connection
  in order to install the libraries required to get WiFi working. For this reason we recommend installing Ubuntu Desktop for the purposes of this assignment. 

#### NVIDIA Jetson Nano Developer Kit (2GB and 4GB)
- Setup instructions can be found [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit) for the 4GB and [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit#prepare) for the 2GB.
- To get PyTorch working on these devices you will need to use the custom image provided by NVIDIA, which can be found [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write)
  for the 4GB and [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit#prepare) for the 2GB.

### SSH and WiFi
To connect to WiFi on campus, you will need to register the device's MAC address in order to connect to the CMU-DEVICE network. 
You can find instructions [here](https://www.cmu.edu/computing/services/endpoint/network-access/wireless/how-to/cmudevice.html).
It's possible the device has already been registered. If you're already able to connect to CMU-DEVICE, then the device is already registered.
You will then be able to connect to the device when you are on the CMU network (for example, from your laptop connected to CMU-SECURE.)
You can find the device's static IP [here](https://getonline.cmu.edu/hosts/manage/).

If you want to SSH in to the device from outside the CMU-SECURE network (e.g. from home without using the CMU VPN), then you will need to get a public IP assigned to the device. 
You can email Computing Services (it-help@cmu.edu) to request this.

It might take a few hours for the registration process on CMU-DEVICE, and about a day turnaround to get a public IP from Computing Services.

If you choose to keep the device at home, then you will need to setup port-forwarding on your home router. 
The router should send all port 22 traffic to the device, and you will ssh through the public IP which can be found in the router settings.

You will also need to enable ssh access to the device. By default, ssh will be disabled.

0. Report the device you are using, and the operating system you're running on it.
1. Make sure PyTorch is installed on the device. Report the version of PyTorch you're using. You can use the [`collect_env.py` script](https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py) to check whether PyTorch is installed and thorough version reporting.

Quantization on-device
---
2. Re-run static and dynamic quantization experiments from [Lab 2](https://github.com/cmu-odml/cmu-odml.github.io/blob/master/labs/02_quantization.md), using just the "Lab 2" model on MNIST.
   Generate the same table as in Lab 2 and include it in your report. If you find that some operations aren't supported on your device, that's ok, please describe this in detail in the first part of your discussion below.

Pruning on-device
---
3. Re-run iterative magnitude pruning (IMP) experiments from [Lab 3](https://github.com/cmu-odml/cmu-odml.github.io/blob/master/labs/03_pruning.md) with and without rewinding.
   Generate the same plot as in Lab 3 (except with fewer experiments) and include it in your report.

Discussion
---
- Did you come across challenges when setting up the device or trying to run your code on this device? List any stumbling blocks, and desribe how you surmounted them, or tried to. 
  Did you find that some operations aren't supported
  Describe one notable thing that you learned by porting your code to this new device.
- Choose either quantization or pruning, and compare and contrast your results from this lab and the previous lab. (Re-)report the hardware you used for the previous lab. 
  Are there trends that you expected, or didn't expect? For example, do latency and space on disk correspond to your expectations, why or why not? 
  Explain what you observed. **You should refer to specific capabilities of the hardware** (such as whether it supports certain operations in certain precisions, or the extent to which it supports vector operations.) 
  
In answering all of the questions in the above discussion, you will likely need to do some research to identify the cause for your findings. 
You may want to start [here](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/) or [here](https://developer.arm.com/Architectures/A-Profile%20Architecture) for Raspberry Pi, and [here](https://developer.nvidia.com/embedded/downloads#?search=Jetson%20Nano) for Jetson Nano.

Grading and submission (10 points)
----
Submit your answers to all the above questions to Canvas as a write-up in pdf format. This assignment is worth 10 points 
(or 10% of your final grade for the class), distributed as follows: 
- **Submission [2 points]:** Assignment is submitted on time.
- **Basic requirements [5 points]:** Answers to all the questions (including all requested tables) are included in the write-up. 
- **Report [2 points]:** Report is well-written and organized with minimal typos and formatting mistakes, and answers to all requested questions are easy to find. Tables are readable and well-labeled. Your results should be easily reproducible from the details included in your report.
- **Thoughtful discussion [1 points]:** The discussion of results in your write-up is thoughtful: it indicates meaningful engagement with the exercise and it is clear that you put effort into thinking through and explaining your results.
