# CancerProject

Main Goal of this project is to predict wheter given person has cancer or not based on below parameters.

Ten real-valued features are computed for each cell nucleus:

1. radius (mean of distances from center to points on the perimeter)
2. texture (standard deviation of gray-scale values)
3. perimeter
4. area
5. smoothness (local variation in radius lengths)
6. compactness (perimeter^2 / area - 1.0)
7. concavity (severity of concave portions of the contour)
8. concave points (number of concave portions of the contour)
9. symmetry
10. fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.


# How to run the app in VS code

In order to run this application write below command in VS Code terminal:

~~~
python application.py
~~~
Address to run this application : [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

# Project UI

Screenshot 1: 
![UI1](./UserInterface/Screenshot1.jpg)

Screenshot 2 (Submit Button):
![UI2](./UserInterface/Screenshot2.jpg)

Output Page Shows Prediction and Probability as below:
![Output](./UserInterface/Output.jpg)

# Best Model Selected

1. Selected Logistic Regression because of Highest test F1 Score of 98.60%
2. 5 fold Cross Validated F1 Score on training is also 98.00%
3. Hence Logistic Regression is best for this data
