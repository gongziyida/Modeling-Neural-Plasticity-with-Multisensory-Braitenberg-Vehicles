# Modeling Neural Development with Multisensory Braitenberg Vehicles
## Content
<strong> 
  
1. Introduction 
2. BV Components

    - Olfactory System
    - Gustatory System
    - Associative Memory
    - Motor Unit
    - Judgement Unit
3. Environment
4. Technical Supports
5. Future Plan
6. Possible Implementation
7. Acknowledgement
8. Reference

</strong>

## 1. Introduction
The project is to create a robust and efficient simulation of Hebbian neural development during cognition processes. 

The simulation utilized a [Braitenberg vehicle (BV)](https://en.wikipedia.org/wiki/Braitenberg_vehicle) that possesses an olfactory system (smell), a gustatory system (taste), an associative memory, a motor unit, and a judgement unit. The BV is allowed to explore freely in an enviornment where sources of olfactory and gustatory stimuli are distributed. 

During its exploration, the BV associates taste with smell when both taste and smell are sensible; when there is no taste, it recalls the taste based on its associative memory and smell received. Both sensed taste and recalled taste can produce preference that affects the BV's movement. Therefore, when the BV becomes more and more mature via association, it can avoid the unpleasant and approach the pleasant, like small animals.

## 2. BV Components
### 2.1. Olfactory System
The olfactory system, in general, is implemented as a Li-Hopfield network, firstly proposed by Li and Hopfield in 1989 and is now still used for modeling [olfactory bulb](https://en.wikipedia.org/wiki/Olfactory_bulb). A Li-Hopfield models the dynamics of the two most important cells in olfactory bulb: mitral cells and granual cells. Mitral cells are both stimulus receivers and delievers (i.e. they are responsible for both input and output) while granual cells are inhibitors to mitral cells. In reality, there are much more granual cells than mitral cells, but in this model, they are equal in amount for simplicity. In addition, mitral cells are inter-connected, therefore forming a circle or a sphere.

<img src="/img/olf.png" alt="Li-Hopfield" width="310"/>

The grey dots are the mitral cells and the black are the granual cells. Red means excitation and blue means inhibition.

The Li-Hopfield network can thus be seen as a group of coupled non-linear oscillators. In short, it is able to alter its oscillatory frequencies based on changes in olfactory attributes, so it is important to "filter" the noise and identify which stimulus source the BV is approaching. The signal powers of the output are then calculated, instead of modeling a complexed afferent nerve in real nervous system.

The olfactory system is implemented in `Layers.LiHopfield` class.

### 2.2. Gustatory System
Gustatory system is only a single layer, for taste is simply "impression" in this simulation. There is no noise involved in taste, or any other perturbation, so further processing of taste is redundant.

The gustatory system is implemented in `Layers.Single` class.

### 2.3. Associative Memory
The associative memory, implemented as a [bidirectional associative memory (BAM)](https://en.wikipedia.org/wiki/Bidirectional_associative_memory), is where Hebbian learning happens. Rather than Hebbian rule that BAM often utilized, [Generalized Hebbian algorithm](https://en.wikipedia.org/wiki/Generalized_Hebbian_Algorithm) is used, for it is demonstrably stable. 
<img src="/img/asso.png" alt="BAM" width="310"/>

Notably, the learning rate should be properly set and converge to zero for accuracy, but in this simulation it is a small constant, because this learning process, unlike typical machine learning with neural networks where samples are learned one by one, occurs in a space where samples are mixed and the times learning each sample is unknown (so are the orders of learning samples). It is hard to determine the initial learning rate and control the converging pace. It is also hard to avoid the effect of initial conditions if the convergence is introduced. 

Moreover, a depression function <img src="/formula/dep_function.png" alt="Depression function" width="150"/> is used to cancel the effect of repeatedly learning from one stimulus source and noisy data. Its positive effect was deonstrated through static testing where the BV does not move and stimuli are just feeded, yet not demoonstrated in the actual simulation

The associative memory is implemented in `Layers.BAM` class.

### 2.4. Motor Unit
The motor unit is radian-based. The BV moves along the heading direction whose value is between negative pi and positive pi.  When the increase in preference passes a threshold, the BV moves forward with a little offset based on the increase; when the decrease in preference passes the threshold, the BV moves backward with a little offest based on the decrease. Otherwise, it moves towards a nearby source.

The motor unit is implemented in `Movement.RadMotor` class.

### 2.5. Judgement Unit
An array of preference function should be defined before intializing `Simulation` class. The preference, the output of the judgement unit, is the sum of the output of each preference functions applied to their corresponding gustatory attributes.

The judgement unit is incorporated in `Simulation` class.

## 3. Environment
The enviornment is realistic that olfactory stimuli decay with distances exponentially from their sources, while gustatory stimuli are sensible only when the BV is close to their sources.

<img src="/img/odor_space_0.png" alt="Odor" width="300"/> <img src="/img/taste_space_0.png" alt="Taste" width="300"/> 

Odor space of one olfactory attribute (left) and taste space of one gustatory attribute (right).

The mapping between olfactory attributes and gustatory attributes should be defined before intializing `Space.Space` class and `Simulation` class.

## 4. Technical Supports
This project uses Cython and C. The most time consuming parts are either written in `core.c` or implemented by using [BLAS functions](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) that Numpy uses. Static images such as those shown above are produced through [Networkx](https://networkx.github.io/) and Matplotlib, while real-time animation is generated using [PyQtGraph](http://pyqtgraph.org/). 

<img src="/img/real_time_animation_example.png" alt="Animation example" width="600"/>

An example of real-time animation.

In addition, experiments were conducted using IPython and Jupyter-Notebook.

## 5. Future Plan
- Put the trained BV in a new, testing environment, like conducting animal testings.
  - Implement the progress saving functionality

## 6. Possible Implementation
- More complexed senses or more than two senses
- More than 1 BV and BVs interactions
- Another possible solution: apply genetic algorithm or other kinds to optimize the network structure

## 7. Acknowledgement
Thank Dr. Bradly Alicea for mentoring and holding weekly meetings. Also thank Stefan Dvoretskii, Jessse Parent, and Ankit Gupta for giving suggestions.

Thanks [Google Summer of Code](https://summerofcode.withgoogle.com) and [INCF](https://www.incf.org) for opening the topic of modeling neural development, and reviewing my proposal on that topic.

## 8. Reference
Li, Z. & Hopfield, J. J. (1989). Modeling the Olfactory Bulb and its Neural Oscillatory Processings. *BioI. Cybern.* 61, 379-392.

Nagayama, S., Enerva, A., Fletcher, M. L., Masurkar, A. V., Igarashi, K. M., Mori, K., & Chen, W. R. (2010). Differential axonal projection of mitral and tufted cells in the mouse main olfactory system. *Frontiers in Neural Circuits* 4(120). doi: 10.3389/fncir.2010.00120.

Sanger, T. D. (1989). Optimal Unsupervised Learning in a Single-Layer Linear Feedforward Neural Network. *Neural Networks* 2, 459-473.

Smith, D. V. & St John, S. J. (1999). Neural coding of gustatory information. *Current Opinion in Neurobiology* 9, 427-435. 

Shepherd, G. M., Chen, W. R., Willhite, D., Migliore, M., & Greer C. A. (2007). The olfactory granule cell: From classical enigma to central role in olfactory processing. *Brain Research Reviews* 55, 373-382.

Soh, Z., Nishikawa, S., Kurita, Y., Takiguchi, N., & Tsuji, T. (2016). A Mathematical Model of the Olfactory Bulb for the Selective Adaptation Mechanism in the Rodent Olfactory System. *PLoS ONE* 11(12), e0165230. doi: 10.1371/journal. 

Wu, A., Dvoryanchikov, G., Pereira, E., Chaudhari, N., & Roper, S. D. (2015). Breadth of tuning in taste afferent neurons varies with stimulus strength. *Nat. Commun.*  6:8171. doi: 10.1038/ncomms9171.
