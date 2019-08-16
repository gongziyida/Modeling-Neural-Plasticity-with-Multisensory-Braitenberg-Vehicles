# Modeling Neural Development with Multisensory Braitenberg Vehicles

## Introduction
The project is to create a robust and efficient simulation of Hebbian neural development during cognition processes. 

The simulation utilized a [Braitenberg vehicle (BV)](https://en.wikipedia.org/wiki/Braitenberg_vehicle) that possesses an olfactory system (smell), a gustatory system (taste), a motor unit, an associative memory, and a judgement unit. The BV is allowed to explore freely in an enviornment where sources of olfactory and gustatory stimuli are distributed. The enviornment is realistic that olfactory stimuli decay with distances exponentially from their sources, while gustatory stimuli are sensible only when the BV is close to their sources.
![Olf example](/img/odor_space_0.png) ![Gus example](/img/taste_space_0.png) 

During its exploration, the BV associates taste with smell when both taste and smell are sensible; when there is no taste, it recalls the taste based on its associative memory and smell received. Both sensed taste and recalled taste can produce preference that affects the BV's movement. Therefore, when the BV becomes more and more mature via association, it can avoid the unpleasant and approach the pleasant, like small animals.

## BV Components
### Olfactory System
The olfactory system, in general, is implemented as a Li-Hopfield network, firstly proposed by Li and Hopfield in 1989 and is now still used for modeling [olfactory bulb](https://en.wikipedia.org/wiki/Olfactory_bulb). A Li-Hopfield models the dynamics of the two most common and most important cells in olfactory bulb: mitral cells and granual cells. Mitral cells are both stimulus receivers and delievers (i.e. they are responsible for both input and output) while granual cells are inhibitors to mitral cells. In reality, receiving excitatory input from mitral cells and giving out inhibitory output, granual cells are much more than mitral cells, but in this model, they have equal amounts for simplicity. In addition, mitral cells are inter-connected, therefore forming a circle or a sphere.
![Li-Hopfield network](/img/olf.png)
The grey dots are the mitral cells and the black are the granual cells. Red means excitation and blue means inhibition (the same for the following diagrams).

The Li-Hopfield network can thus be seen as a group of coupled non-linear oscillators. In short, it is able to alter its oscillatory frequencies based on changes in olfactory attributes, so it is important to "filter" the noise and identify which stimulus source the BV is approaching.

The signal powers of the output are then calculated, instead of modeling a complexed afferent nerve in real nervous system.

### Gustatory System
Gustatory system is only a single layer, for taste is simply "impression" in this simulation.
