## Project statement

We wish to explore different machine learning (ML) models for readout predictions of a supercondicting qubit. This is done by simulating readout data with a quantum mechanics model (QM) to train the ML model. We aim to implement our physical intuition of the quantum mechanical system and ultimately interpret the readout with fewer datapoints and with a higher efficiency. We assume that the readout will converge to some state, and that it can be time dependent. 

### Background & toy model: (conceptual knowledge level)
- Interpret generalized meaurements of quantum systems as partial collapse of a coupled to a meter system 
- Simulate experiments with weak measurement of the von Neumann model with a quantum random walk (simple open quantum system without enviroment interaction)
- Create a simple artificial neural network (eg. CNN) and a bayesian machine learning model to predict steady state solution of the state in the von Neumann model. This can be both with flip rate utilising binary classification, and non-flip rate utilising seq-to-seq models. If time allows predict the steay state solution analytically too. 
- Explain the physical meaning of the measurement results and the ML models explainability. 
- Assess the limitation that stems from the simplification in the von neuman model and in the black box architecture in the ANN model. 
- Keeping this in mind, plan what simplifications we can adress and what QM & ML model to continue with (include new assumptions). 

### Further work (procedual knowledge level)
- Predict the impact the new assumption has for a new QM model (leaking information to the enviroment, mixed states and trace out subsystems ---> Stochastic master equation). 
- Do the same for the ML model (probability distribtution for estimators ---> sequence to sequence bayesian machine learning)
- Construct a new 'equation of motion' for the QM model for a superconducting qubit with homodyne detection. Simulate the experiment. Predict the state only using the simulated readout.
- Construct a new design for the ML model. Predict the state of the system using the simulated readout.
- Recognize what differences there are in the prediction proces for the ANN and baysian machine learning model on both the toy model and the superconducting qubit model. In order to do so, implement the ANN on the superconducting qubit too. 
- Conclude how succesful our baysian machine learning models is to predict the steady state for both QM models (compared to our ANN or analytic).

