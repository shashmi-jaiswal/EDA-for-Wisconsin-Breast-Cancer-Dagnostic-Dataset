# Breast-Cancer-Dataset-Diagnostic-
**Description**:

Breast cancer has been seriously threatening both the physical and mental health of women in the world. It is one of the most malignant cancers however its curable in an early-stage, non-metastatic disease. This underlies the importance of early detection and need to conduct research in these lines. The Breast Cancer Wisconsin Dataset gives an opportunity to apply Machine learning techniques in such a research process.  An important step before deploying any Machine Learning algorithm is to study the characteristics of the given dataset, commonly called Exploratory Data Analysis. This blog will help us understand this dataset in detail using basic Python code.

**Breast Cancer Wisconsin (Diagnostic) Dataset**. 

This dataset is available to download from the Kaggle website at . This can also be downloaded from the UCI Machine Learning Repository at. 
Diagnosis of breast cancer is traditionally done by full biopsy which is an invasive surgical method. A less invasive method is Fine Needle Biopsy (FNB) which allows for examination of a small amount of tissue from the tumor. This dataset was obtained by analyzing the cell nuclei characteristics of 569 images obtained by Fine Needle Aspiration of the breast mass. Each of the images are classified(diagnosed) as being “Benign” or “Malignant”
Benign: 
Malignant:
Ten real-valued features are computed for each cell nucleus:
1.	radius (mean of distances from center to points on the perimeter)
2.	texture (standard deviation of gray-scale values)
3.	perimeter
4.	area
5.	smoothness (local variation in radius lengths)
6.	compactness (perimeter^2 / area - 1.0)
7.	concavity (severity of concave portions of the contour)
8.	concave points (number of concave portions of the contour)
9.	symmetry
10.	fractal dimension ("coastline approximation" - 1)
The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
All feature values are recoded with four significant digits.

**Data Preprocessing:**
The column "diagnosis" has two values: Malignant and Benign.
Machine learning models can be built on data that is made of just numbers. Hence, we will replace Malignant with number 1 and Benign with number 0. Any two numbers can be used but 0 and 1 are the most commonly used for classification purposes.
Once replaced, the code df.diagnosis.unique()  will serve as a check to that we get a resulting column of numbers 1 and 0

**Data Analysis**


**1. Unique Values:**
Lets look at the numnber of unique values we have in each column/feature. Since this data is numberical we may expect to find a lot of variable data (meaning not same values)

**2. Data Spread between the two types**
One way to look is acheive this it to count the total number of 1's in the diagnosis column. (1 represents the malignant data point), Getting the mean of the '"diagnosis" column and multiplying it by 100  will give the percentage of data points that belong to the malignant type.

**3. Correlation Matrix:**
One of the assumptions in most of the key Machine learning models is that no variable in the model is highy correlated to any other variable. A high correlation between variables causes the problem of 'multicollinearity' and hence its important to be aware of the relationsships between each variable to better interpret the results of a model.  

Pearsons Correlation Coeffecient:
One way to quantify the relationship between given two variables is a Pearson correlation coeffecient which is a measure of linear relation and has a values betweeen -1 and 1. 

    -1 indicates a perfectly negative linear correlation between two variables

    0 indicates no linear correlation between two variables

    1 indicates a perfectly positive linear correlation between two variables

Correlation Matrix:
A coorelation matrix shows the relation between two given variables in the form of a matrix. Sucha a matrix colored with a heat map will make it much easier to read.
For this data, we will use the sklearn implentation of correlattion matrix which uses the Pearson Correlation Coeffecient as its default. 

Diagnoal Correlation Matrix:
Since the correlation matrix is symmetrical, half of the coeffecients are redundant. So, a diagnoal correlation matrix can be used for an quick and easy viewing. 


Reference: https://www.statology.org/how-to-read-a-correlation-matrix/

Though, I personally like this plot but interpreting such a plot can be tedious when we have multiple variable(features) and we move the plot back and forth to compare all values. So another way to view such plots is to plot a diagonal correlation matrix. But the inferences from both are esentially the same. 

**Obervations from Correlation Matrix:**
In a corrleation hat map, the way its interpreted is higher the corrlelation value, the more correlated the two variables(features) are
1. Radius, Area and Permiter are correlated (corr>0.9) which is obvious as area and perimeter is calculated using the radius values.

2. Texture_mean and texture_worst are higly correlated with corr_value = 0.98 (texture_worst is the largest value of all the textures)

3. Compactness_mean,concavity_mean,concave_points_mean are also highy correlated with values in range 0.7 to 0.9

4. Symmetry_mean and symmetry_worst are correlated too by values 0.7

5. Fractural_dimension_mean and fractural_dimension_worst are correlated by value 0.77

Lets analyse these features in more depth by looking at distribution plots of each of these features. We could look at them in the same grouped order to keep our thoughts in flow.

**4. Pair Plots:**
Pair plots allows us to see both distribution of single variables and relation between a pair of variables from the given set of variables/features in a form of a matrix

The key reason to use a pair of plots is that humans have a capacity to understand relation between two variables using single plot like a 2D scatter plot or a KDE plot. One can use 3D plots to visualize relationship between 3 given variables. But real-world data, as in our case, has more than 3 variables. So a good approach is to study relationships between a given pair of features. Seaborn's pairplot feature comes in handy in such cases and is widely used. 


Creating a pair plot with 30 features is time consuming. So, we will use just the "mean" values of the given feartures to get an idea of the distributions. 
  
 **5. Density plots:**

These are smoothened versions of a histogram which plots the values of a feature/column in equally binned distrubutions and then smoothens it using kernel smoothing to create a well defined distribution shape.
  **Obeservations**
  The diagnol plots are the kde plots for each varible/feature. The upper and the lower triangle are essentially mirror images of each other.

  One major take-away from these plots to observe that there is a certian level of separation between the malignant and the benign types which can make it helpful to use these features to design a ML alogorithm

**Violin Plots and Box Plots:**
Box Plot: A box plot gives a information about the variabibnlity and dispersion of the data in the form of a box
THe middle line respresents teh median value and with the first and third quarltile on either ends enges. The Minimum and the Maximum are represented by lines on either side of the box. THe 
Violin Plots has the advantage of comibing this data with its density plots hencle allowing us to see the entirity of the distrubution of the data in one plot. 

A combination of all these plots will help visualize the distributions of the data whihch act as important tools to help design the features to build a machine learning algorithms. This also helps in interpretiing the results of a given alogorithm.

We will use the sklearn and seaborn for creating and visualization of each of these plots.

**6.Removing Outliers - Local Outlier Factor:**
  Local Outlier Factor

  Assuming, that the inlier data is Gaussian distributed, We will use Local Outlier Factor as a way to detect the outlier in our given dataset.

  The Local Outlier Factor is an unsupervised anamoly detection method that computes a score called Local outlier factor(LOF). LOF measures the local density deviation of a given datapoint with respect to its k- number of neighbors.

  The LOF is calulated as a ratio of the average local desity of its k-nearest neighbors and its own local neighbor. Any inlier will have the same local density as its neigbors but an outlier will have a much lower local density.
