## Forest Fire Linear Regression in R

Linear Regression and hyper parameter tuning project using the Forest Fire dataset from UCI Machine Learning Repository [https://archive.ics.uci.edu/ml/index.php]

The variables are as follows: X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain, and area.  

- X is the x-axis coordinates from the Montesinho park map  
- Y is the y-axis coordinates from the Motesinho park map  
- month is the most of the year  
- day is the day of the week  
- FFMC is the index of FFMC from the FWI system  
- DMC is the index of DMC from the FWI system  
- Dc is the index of DC from the FWI system  
- ISI is the index of ISI from the FWI system  
- temp is the temperature in degrees Celsius  
- RH is the relative humidty in percentage  
- wind is the windspeed in km/h  
- rain is the amount of rain in mm  
- area is the amount of burned area in ha  

[Cortez and Morais, 2007] P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data. In J. Neves, M. F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, Guimarães, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9. Available at: [https://archive.ics.uci.edu/ml/datasets/Forest+Fires]

# Summary
In this project, I analyzed the forest fire dataset from UCI learning repository using linear regression, random forests and gbm. The reason why I carried out this project is I wanted to find out the best machine learning method to use on this dataset.  

I began the project by cleaning the data so that it was suitable for use. I first started by removing any NA values within the dataframe. From there, I performed a logarithmic transformation on the column area because the data was skewed towards zero. In order to prevent errors in the logarithmic transformation, I added one to each observation so that there would not be any logarithmic transformation of the number zero which leads to a computational error as log10(0) does not exist. I finally read in the categorical variables month, and day as a numeric factor and now the dataset is ready for use.
```
ff=read.csv("forestfires.csv")
ff=ff%>%drop_na()
ff$area=log10(ff$area+1)
ff$month=as.numeric(as.factor(ff$month))
ff$day=as.numeric(as.factor(ff$day))
```

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
I compared the RMSE of four different types of models: full model linear regression, best subset (stepwise) regression, random forest, and gbm. The best subset used X, DMC, RH, and wind to predict the response variable. For the random forest method, and the gbm method, I started off by using the default parameters so that there wasn’t any bias in the results.

```
K=10; N = nrow(ff)
folds = get.folds(N,K)

#Creating Dataframe
my_rmse = array(0,dim=c(4,6))
rownames(my_rmse) = c("linear_mod","best_subset","rf","gbm")
colnames(my_rmse)=c("1","2","3","4","5","Mean")

for(i in 1:5){#for k folds
  
  train_set = ff[folds!=i,]
  test_set = ff[folds==i,]
  test_set_validation=test_set$area 
  
  #Full linear model
  linear_reg = lm(area~.,train_set)
  linear_preds = predict.lm(linear_reg,test_set)
  my_rmse["linear_mod",i] = mean((test_set_validation-linear_preds)^2)
  
  #Best subset linear model
  best_sub=step.model=stepAIC(linear_reg,direction="both",trace=FALSE)
  best_preds=predict.lm(best_sub,test_set)
  my_rmse["best_subset",i]=mean((test_set_validation-best_preds)^2)
  
  #Rf
  random_f=randomForest(area~.,data=train_set,importance=TRUE,ntree=500)
  rf_preds=predict(random_f,test_set)
  my_rmse["rf",i]=mean((test_set_validation-rf_preds)^2)
  
  #gbm
  gbm=gbm(area~.,distribution = "gaussian",data=train_set)
  gbm_preds=predict.gbm(gbm,test_set)
  my_rmse["gbm",i]=mean((test_set_validation-gbm_preds)^2)
```

![image](https://user-images.githubusercontent.com/95319198/144773367-7ae05efe-070d-4f44-a4d1-a5548321d183.png)  

From the results, we can see that gbm performed the best with the lowest RMSE after taking the average of the iterations. From there, for the random forests and stochastic gradient boosting methods, I tuned the parameters so that I knew which parameters provided the best results. 
```
rfGrid <-  expand.grid(mtry = c(2,3,4,5,6,7,8,9,10,11))
rfControl <- trainControl(method = "cv",number = 10)
rf_Fit <- train(area ~ ., data = train_set1, method = "rf", n.trees=500)
rf_Fit

gbmGrid=expand.grid(n.trees=c(300,400,500,600,800,1000),interaction.depth=c(1,2,3),shrinkage=c(0.001,0.01,0.05,0.25),n.minobsinnode=c(10))
gbmControl <- trainControl(method = "cv",number = 10)
gbm_Fit <- train(area ~ ., data = train_set1, method = "gbm", verbose=FALSE)
gbm_Fit
```
The resulting best models are as follows:   
![image](https://user-images.githubusercontent.com/95319198/144773328-ec99751d-b2c0-49bd-a1c2-d0c3e92bcb9e.png)  

![image](https://user-images.githubusercontent.com/95319198/144773349-a7ef58a3-3b67-4dbd-a32e-fb1ae3f925ca.png)  

I then tested the gbm method and the random forest methods with the optimal values for their adjustable parameters. I once again compared the RMSE of the newly improved models and the results are now as follows.  

```
best_gbm=gbm(area~.,distribution = "gaussian",data=train_set1,n.trees=50,interaction.depth=1,n.minobsinnode=10,shrinkage=0.1)
for(i in 1:5){
  #Rf
  best_rf=randomForest(area~.,data=train_set1,importance=TRUE,ntree=500,mtry=2)
  best_rf_preds=predict(best_rf,test_set1)
  my_rmse1["rf_tuned",i]=mean((test_set_validation1-best_rf_preds)^2)
  #GBM
  best_gbm=gbm(area~.,distribution = "gaussian",data=train_set1,n.trees=50,interaction.depth=1,n.minobsinnode=10,shrinkage=0.1)
  best_gbm_preds=predict.gbm(best_gbm,test_set1)
  best_rf_preds=predict(best_rf,test_set1)
  my_rmse1["gbm_tuned",i]=mean((test_set_validation1-best_gbm_preds)^2)
}
For the mean of results
for(i in 1:nrow(my_rmse1)){
  my_rmse1[i,6]=mean(my_rmse1[i,1:5])
}

my_rmse1
```
![image](https://user-images.githubusercontent.com/95319198/144773428-bc3dd63e-84f6-4dd1-b683-d85b5ab97e82.png)  
Overall, the methods: linear regression, best_subset, random forest all had similar RMSE values all around 0.40. Gbm with the default parameters and random forests with the tuned parameters had a slightly better RMSE of 0.39. The gbm method with tuned parameters had the best results with an average RMSE of 0.3386348. There are many more machine learning techniques, but in this project, of the six tested models, the gbm method with tuned parameters had by far the best fit.  






