# Forest Fire Linear Regression in R

Linear Regression and hyper parameter tuning project using the Forest Fire dataset from UCI Machine Learning Repository [https://archive.ics.uci.edu/ml/index.php]

The variables are as follows: X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain, and area.  

X is the x-axis coordinates from the Montesinho park map  
Y is the y-axis coordinates from the Motesinho park map  
month is the most of the year  
day is the day of the week  
FFMC is the index of FFMC from the FWI system  
DMC is the index of DMC from the FWI system  
Dc is the index of DC from the FWI system  
ISI is the index of ISI from the FWI system  
temp is the temperature in degrees Celsius  
RH is the relative humidty in percentage  
wind is the windspeed in km/h  
rain is the amount of rain in mm  
area is the amount of burned area in ha  

[Cortez and Morais, 2007] P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data. In J. Neves, M. F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, Guimarães, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9. Available at: [https://archive.ics.uci.edu/ml/datasets/Forest+Fires]

In this project, I analyzed the forest fire dataset from UCI learning repository using linear regression, random forests and gbm. The reason why I carried out this project is I wanted to find out the best machine learning method to use on this dataset.

I began the project by cleaning the data so that it was suitable for use. I first started by removing any NA values within the dataframe. From there, I performed a logarithmic transformation on the column area because the data was skewed towards zero. In order to prevent errors in the logarithmic transformation, I added one to each observation so that there would not be any logarithmic transformation of the number zero which leads to a computational error as log10(0) does not exist. I finally read in the categorical variables month, and day as a numeric factor and now the dataset is ready for use.  

This is a histogram of area before taking a logarithmic transformation. It is clearly skewed towards zero.  
![image](https://user-images.githubusercontent.com/95319198/144772914-b7315805-6db8-45e5-b586-90e5600c4f88.png)  


This is a histogram of area after taking a logarithmic transformation.
![image](https://user-images.githubusercontent.com/95319198/144772930-5f5122bc-7544-4c86-b910-2db3083d60dc.png)  

I created my own kfolds splitting function and then utilized it so that I could compare the different models efficiently. 
```
get.folds = function(n, K) {
n.fold = ceiling(n / K) 
fold.ids.raw = rep(1:K, times = n.fold)
fold.ids = fold.ids.raw[1:n]
folds.rand = fold.ids[sample.int(n)]
return(folds.rand)
}
```
I compared the RMSE of four different types of models: full model linear regression, best subset (stepwise) regression, random forest, and gbm. The best subset used X, DMC, RH, and wind to predict the response variable. For the random forest method, and the gbm method, I started off by using the default parameters so that there wasn’t any bias in the results. The results are as follows:


