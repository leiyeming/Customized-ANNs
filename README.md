# Customized ANNs for CPUE Standardization
This repository implements Customized Artificial Neural Networks (ANNs) for CPUE (Catch Per Unit Effort) standardization in fisheries management. Our models address overfitting and interpretability challenges, providing improved accuracy in fish abundance trends (standardized CPUE) and distribution patterns. Visualizations of fish distribution and fishing locations for all scenarios are available in the "additional figures" directory.
Features

- ANNs tailored to specific fisheries and ecological systems
- Spatial-temporal modules for dynamic fish distribution modeling
- Enhanced interpretability of underlying relationships
- Overfitting mitigation for better model generalization

We introduce two customized ANN models - ANN S and ANN ST - designed to improve interpretability and reduce overfitting:

- ANN S: Incorporates a spatial module using longitude and latitude as inputs. The spatial module's output feeds into the final output neuron along with other explanatory variables, allowing nonlinear spatial dependence while assuming consistent fish distribution across years.
- ANN ST: Features a spatial-temporal module accepting longitude, latitude, and year as inputs. This design accounts for potential temporal variations in fish distribution, offering a more dynamic data representation.

Both models combine their respective module outputs (f(lon, lat) for ANN S and f(lon, lat, year) for ANN ST) with other variables to produce the final output. A comparative illustration of these modified ANNs and the original ANN is provided in the figure below.


Figure illustrates various ANN model structures for CPUE standardization:
a) ANN_F: A standard multilayer perceptron (MLP) that uses all variables as direct inputs.
b) ANN_S: Incorporates a dedicated spatial module processing longitude and latitude interactions. The output of this module is then combined with other variables.
c) ANN_ST: Features a spatial-temporal module to capture interactions between location (longitude and latitude) and time.
d) Customized ANN: An example structure incorporating multiple non-linear terms for more complex relationships.
The code directory includes implementations for constructing and running the ANN_ST model (structure c).

![img](Github_page/structures.png)










