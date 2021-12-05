### A NEW INTERFACE FOR DATA ANALYSIS 
<br/>
<div align="center" >
<img width="800" alt="Capture d’écran 2021-12-05 à 16 02 12" src="https://user-images.githubusercontent.com/31353252/144753986-b5550b3c-3384-4470-92d8-81bdb7b694f9.png">
</div>
<br/>

### DESCRIPTION

The overall objective of this project is to develop a graphical interface to apply prediction algorithms on a well-defined dataset. This application will be intended for users who do not have sufficient skills in Python programming to apply machine learning algorithms. 
The tool focuses on supervised learning algorithms, including classifications and regressions. It allows users to apply these algorithms to the data, and provides as output the evaluation metrics and especially the graphical representations adapted to each model.

### OBJECTIFS SPECIFIQUES

As specific objectives, the following modules will be developed: <br/><br/>
**Back End (Functionality)**
In this part, we implement the main features of the application (functional requirements), that is to say the different operations that the user will be able to do with the tool. In summary, we will allow the user to
to load his data
 * to visualize his data and to see the descriptive statistics (correlation, projection in the plane,...)
 * choose the prediction algorithm to apply on the data:
 * Classification: Linear Discriminant Analysis, Decision Tree, Logistic Regression
 * Regression: Simple/multiple linear regression, Random forest, KNN
 * Results display: metrics, cross-validation, graph, etc.<br/><br/>
**Front End (Graphical Interface)**
 * Creation of the graphical interface with bokeh.

### Architecture
<div align="center">
  <img width="800" alt="Capture d’écran 2021-12-05 à 16 14 01" src="https://user-images.githubusercontent.com/31353252/144754341-d27943b9-1169-4a62-9d51-15b0ce9a7569.png">
</div>

<br/>
This architecture explains the different messages exchanged between the system and the outside. 
  1. The user interacts with the interface (Front End) to make a request;
  2. From the interface, function calls (Back End) are made;
  3. The Back End performs the operations and returns the result to the interface (Front End) 
  4. The interface returns the result to the user <br/>

We have applied the concepts of object-oriented programming for a better organization of the code. In summary, we have three files :
- **Interface.py** which contains all the components of our application, and therefore constitutes our front end. 
- **Regression.py** which contains our business logic for regression
- **Classification.py** contains the business logic for classification

### User's guide
In this section, we will provide you with a user's guide, presenting the different possibilities offered by our solution:
<br/>
<div align = "center">
<img width="800" alt="Capture d’écran 2021-12-05 à 16 22 36" src="https://user-images.githubusercontent.com/31353252/144754628-d65bb441-c83a-44f0-9452-e2dfef0f5a1d.png">
</div>

Our interface is organized in sections: <br/>
- A first section (**data loading and descriptive statistics**) where we allow the user to load their data and see the descriptive statistics. It is divided into two sub-sections: 
A first sub-section allowing the user to select the parameters necessary to perform the analysis desired by the user. The user can load his data set in .csv or .xlsx format, choose the x and y variables to be correlated together and a projection variable if he wishes, to project the points in the plane.
- The second subsection provides an overview of all exploratory analyses of the data. 

<div align="center">
  <img width="750" alt="Capture d’écran 2021-12-05 à 17 45 26" src="https://user-images.githubusercontent.com/31353252/144757421-1fc0e00b-cffb-47a1-bed9-27c8c271d053.png">
<img width="750" alt="Capture d’écran 2021-12-05 à 17 45 57" src="https://user-images.githubusercontent.com/31353252/144757423-3ba042c7-2596-4661-ae44-75efd73a70fe.png">
<img width="750" alt="Capture d’écran 2021-12-05 à 17 46 18" src="https://user-images.githubusercontent.com/31353252/144757439-e78e2c21-b2af-489f-8025-3e380fe27209.png">
</div>

