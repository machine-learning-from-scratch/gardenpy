![GardenPy Logo](https://github.com/machine-learning-from-scratch/gardenpy/blob/main/docs/gardenpy_flat_logo.png)

------------------------------------------------------------------------------------------------------------------------

GardenPy is a package that integrates an automatic differentiation package with machine learning algorithms to build
machine learning from scratch.

It's built on NumPy, utilizing NumPy's computation to calculate linear algebra results.
It utilizes similar syntax, meaning most forward-pass math in NumPy applies to GardenPy's Tensors, with the
added benefit of automatically differentiating Tensors.

There are many machine learning packages, many of which are faster and easier to use than this one.
This package's purpose isn't to compete with the speed or features of other packages; instead, the package's goal is
to allow the creation of machine learning models using syntax as close to mathematical syntax as possible, to allow the
model's creator to truly understand the mathematics behind machine learning.

------------------------------------------------------------------------------------------------------------------------

## **Mathematical Syntax**

GardenPy's purpose isn't to compete with other machine learning libraries regarding speed or features.
Rather, GardenPy's mission is to allow the creation of machine learning models using syntax that is as similar to mathematical
syntax as possible.

------------------------------------------------------------------------------------------------------------------------

## **Dynamic Computational Graphs**

GardenPy automatically creates a computational graph to represent all calculations.
When a line calls for gradient calculation, GardenPy uses tape-based automatic differentiation and a search tree to
relate two Tensors.

![Computational Graph](https://github.com/machine-learning-from-scratch/gardenpy/blob/main/docs/computational_graph.gif)

------------------------------------------------------------------------------------------------------------------------

## **Expansion of Methods**

GardenPy's automatic differentiation is designed to be easily expanded.

------------------------------------------------------------------------------------------------------------------------

## **Installation**

### **Prerequisites**

This library tries to do as many things as possible without relying on pre-defined methods.
Below are the required prerequisites and what they're used for.

- Python 3.9 +
  - Codebase.
- NumPy 1.26.1 +
  - Linear algebra computations.

To install GardenPy, ...

------------------------------------------------------------------------------------------------------------------------

## **Examples**

GardenPy has pre-built model setups to quickly create models to train.
These models utilize GardenPy's automatic differentiation and can be used as examples of how to best use GardenPy's
automatic differentiation.
Additionally, GardenPy has example applications on the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset and
a simple checkered or non-checkered dataset.

------------------------------------------------------------------------------------------------------------------------

## **Website**

The Machine Learning from Scratch team also offers a [website](http://45.63.57.237/) with mathematics for machine
learning, detailed documentation, diagrams, videos, and an example of a dense neural network trained on the
[MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset. The website's repository can be found [here](https://github.com/machine-learning-from-scratch/NeuralNetWebsite).

------------------------------------------------------------------------------------------------------------------------

## License
**GardenPy** is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.en.html).
See [License](`LICENSE`) for more.

------------------------------------------------------------------------------------------------------------------------

## Contact
For questions, recommendations, or feedback on GardenPy, contact:
- Christian SW Host-Madsen (c.host.madsen25@gmail.com)
- Doyoung Kim (dkim25@punahou.edu)
- Mason YY Morales (mmorales25@punahou.edu)
- Isaac P Verbrugge (isaacverbrugge@gmail.com)
- Derek Yee (dyee25@punahou.edu)

Report any issues to [the issues page](https://github.com/machine-learning-from-scratch/gardenpy/issues).

------------------------------------------------------------------------------------------------------------------------

This project is a work in progress.

Tensor autograd – Done
Tensor base operations – In progress
ML algorithms – Done
Tensor algorithm integration – Done
Computational graph building – Done
Base models – In progress
Data visualization tools – Not started
Example implementation – Not started
Library setup, README, and website – In progress

------------------------------------------------------------------------------------------------------------------------

![Machine Learning from Scratch Logo](https://github.com/machine-learning-from-scratch/gardenpy/blob/main/docs/machine_learning_from_scratch_large_logo.png)
