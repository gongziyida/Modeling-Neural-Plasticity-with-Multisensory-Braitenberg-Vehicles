# Modeling Neural Development with Multisensory Braitenberg Vehicles

## Introduction
The project is to create a robust and efficient simulation of Hebbian neural development during cognition processes. 

The simulation utilized a [Braitenberg vehicle (BV)](https://en.wikipedia.org/wiki/Braitenberg_vehicle) that possesses an olfactory system (smell), a gustatory system (taste), an associative memory, a motor unit, and a judgement unit. The BV is allowed to explore freely in an enviornment where sources of olfactory and gustatory stimuli are distributed. 

During its exploration, the BV associates taste with smell when both taste and smell are sensible; when there is no taste, it recalls the taste based on its associative memory and smell received. Both sensed taste and recalled taste can produce preference that affects the BV's movement. Therefore, when the BV becomes more and more mature via association, it can avoid the unpleasant and approach the pleasant, like small animals.

## BV Components
### Olfactory System
The olfactory system, in general, is implemented as a Li-Hopfield network, firstly proposed by Li and Hopfield in 1989 and is now still used for modeling [olfactory bulb](https://en.wikipedia.org/wiki/Olfactory_bulb). A Li-Hopfield models the dynamics of the two most common and most important cells in olfactory bulb: mitral cells and granual cells. Mitral cells are both stimulus receivers and delievers (i.e. they are responsible for both input and output) while granual cells are inhibitors to mitral cells. In reality, there are much more granual cells than mitral cells, but in this model, they are equal in amount for simplicity. In addition, mitral cells are inter-connected, therefore forming a circle or a sphere.
<img src="/img/olf.png" alt="Li-Hopfield" width="310"/>
The grey dots are the mitral cells and the black are the granual cells. Red means excitation and blue means inhibition.

The Li-Hopfield network can thus be seen as a group of coupled non-linear oscillators. In short, it is able to alter its oscillatory frequencies based on changes in olfactory attributes, so it is important to "filter" the noise and identify which stimulus source the BV is approaching. The signal powers of the output are then calculated, instead of modeling a complexed afferent nerve in real nervous system.

In the implementation, it is in `Layers.LiHopfield` class.

### Gustatory System
Gustatory system is only a single layer, for taste is simply "impression" in this simulation. There is no noise involved in taste, or any other perturbation, so further processing of taste is redundant.

In the implementation, it is in `Layers.Single` class.

### Associative Memory
The associative memory, implemented as a [bidirectional associative memory (BAM)](https://en.wikipedia.org/wiki/Bidirectional_associative_memory), is where Hebbian learning happens. Rather than Hebbian rule that BAM often utilized, [Generalized Hebbian algorithm (GHA) or Sanger's rule] (https://en.wikipedia.org/wiki/Generalized_Hebbian_Algorithm) is used, for it is demonstrably stable. 
<img src="/img/asso.png" alt="BAM" width="310"/>

Notably, the learning rate should be properly set and converge to zero for accuracy, but in this simulation it is a small constant, because this learning process, unlike typical machine learning with neural networks where samples are learned one by one, occurs in a space where samples are mixed and the times learning each sample is unknown (so are the orders of learning samples). It is hard to determine the initial learning rate and control the converging pace. It is also hard to avoid the effect of initial conditions if the convergence is introduced. 

Moreover, a depression function <img src="/formula/dep_function.png" alt="Depression function" width="200"/> is used to cancel the effect of repeatedly learning from one stimulus source and noisy data. Its positive effect was deonstrated through static testing where the BV does not move and stimuli are just feeded, yet not demoonstrated in the actual simulation

In the implementation, it is in `Layers.BAM` class.

### Motor Unit
The motor unit is radian-based. The BV moves along the heading direction whose value is between negative pi and positive pi.  When the increase in preference passes a threshold, the BV moves forward with a little offset based on the increase; when the decrease in preference passes the threshold, the BV moves backward with a little offest based on the decrease. Otherwise, it moves towards a nearby source.

In the implementation, it is in `Movement.RadMotor` class.

### Judgement Unit
An array of preference function should be defined before intializing `Simulation` class. The preference, the output of the judgement unit, is the sum of the output of each preference functions applied to their corresponding gustatory attributes.

It is incorporated in `Simulation` class.

## Environment
The enviornment is realistic that olfactory stimuli decay with distances exponentially from their sources, while gustatory stimuli are sensible only when the BV is close to their sources.

<img src="/img/odor_space_0.png" alt="Odor" width="310"/>
<img src="/img/taste_space_0.png" alt="Taste" width="300"/> 

Odor space of one olfactory attribute (left) and taste space of one gustatory attribute (right).

The mapping between olfactory attributes and gustatory attributes should be defined before intializing `Space.Space` class and `Simulation` class.

## Techinical Supports
This project uses Cython and C. The most time consuming parts are either written in `core.c` or implemented by using [BLAS functions](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) that Numpy uses. Static images such as those shown above are produced through [Networkx](https://networkx.github.io/) and Matplotlib, while real-time animation is generated using [PyQtGraph](http://pyqtgraph.org/). 


An example of real-time animation.

In addition, experiments were conducted using IPython and Jupyter-Notebook.

## Acknowledgement
Thank Dr. Bradly Alicea for mentoring and holding weekly meetings. Also thank Stefan Dvoretskii, Jessse Parent, and Ankit Gupta for helping.

Thanks Google Summer of Code (GSOC) for letting me propose this project.
