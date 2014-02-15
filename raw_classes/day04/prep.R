##Regression Prep
data<-women

x<-seq(1,10,0.1)
cor(x^2,x^3)

#lets have a look for a selection of variables

d<-data.frame(x=x,
              x2=x^2,
              x3=x^3,
              x4=x^4,
              x5=x^5)
cor(d) ##see how they are correlated

d<-data.frame(x=sin(x),y=cos(x))
cor(d) ## choosing different basis functions to give you an orthogonal basis.. 

##lets motivate the use of a polynomial
fit<-lm(formula=weight ~ height,data=data) # build your linear model
summary(fit)
plot(fit) ## we see could be better

##lets looks at the residuals (pointwise error)
residuals(fit)
plot(women$height,women$weight,
     xlab="Height (in inches)",
     ylab="Weight (in pounds)")
abline(fit)

##we see that we may be able to do better.
fit2<-lm(weight~height + I(height^2),data=women) #note the I for use of regular expression
summary(fit2)
residuals(fit2)
lines(women$height,fitted(fit2))

##here we see R squared has gone up and the that the probability that variable for the 
##squared term is not significant is tiny. By eye, the model looks better too!
## Just to confirm, lets show collinearity here:
cf<-coef(fit2) # get the coefficients

d<-data.frame(x=-cf[1]*data$height,
              x2=cf[2]*I(data$height^2))              
cor(d) ##see how they are correlated

#lets try to use poly which will give us orthogonal basis functions.
fit3<-lm(data$weight~poly(data$height,10)) ##large number of functions, which will
##most likely result in overfitting. In this case is simple, so not, however look at the 
##orthogonal function t vaues to see when they stop being significant.
summary(fit3)
lines(women$height,fitted(fit3))

##Cannot show you that the terms are not correlated! Will have to trust that poly will do its job

##Now, lets use some data to demonstrate overfitting.
ar <- as.data.frame(EuStockMarkets)
ar$year<-seq_len(nrow(ar))
ar<-ar[ar$year>1500,]
plot(ar$year,ar$DAX)

fit5<-lm(ar$DAX~poly(ar$year,25)) ##overfit
plot(ar$year,fitted(fit5))

fit6<-lm(ar$DAX~poly(ar$year,3)) ## this is probably more indicative of the trend.
lines(ar$year,fitted(fit6))

## now lets try regularization. May need to install glmnet: install.packages(pkgs="glmnet")
library(glmnet)
help(glmnet)

ar <- as.data.frame(EuStockMarkets)

#populate x matrix with all the factors from 
ar$year<-seq_len(nrow(ar))
x=matrix(ar$year,ncol=1)
x<-cbind(x,I(ar$year^2))
x<-cbind(x,I(ar$year^3))
x<-cbind(x,I(ar$year^4))
x<-cbind(x,I(ar$year^5))
x<-cbind(x,I(ar$year^6))
x<-cbind(x,I(ar$year^7))


##lasso is l1 norm, in glmnet \alpha=1 (Default) for this.
fit7<-glmnet(x,ar$DAX,family="gaussian")
results<-predict(fit7,newx=x,type="response")

##plot for all lambda (it comes up with its own lambda sequence through search)
plot(ar$year,ar$DAX)
for (i in 1:ncol(results)){  
  lines(ar$year,results[,i])
}

#lambda is the strength of the penalty
cv.glmmod<-cv.glmnet(x,ar$DAX,alpha=1)
plot(cv.glmmod)
best_lambda <-cv.glmmod$lambda.min
results<-predict(fit7,newx=x,s=best_lambda,type="response") #lambda best=1.899118 (low penalisation)
plot(ar$year,results)
lines(ar$year,ar$DAX)

##just to check, lets take the worst one. Recall the error graph above
#e^7=1096.63. Massive penalisation, no generalisation
results<-predict(fit7,newx=x,s=exp(7),type="response")
plot(ar$year,results)
lines(ar$year,ar$DAX)

#e^6 # not great
results<-predict(fit7,newx=x,s=exp(6),type="response")
plot(ar$year,results)
lines(ar$year,ar$DAX)

#e^5 ##should be getting better
results<-predict(fit7,newx=x,s=exp(5),type="response")
plot(ar$year,results)
lines(ar$year,ar$DAX)

#e^5 ##should be getting better
results<-predict(fit7,newx=x,s=exp(2),type="response")
plot(ar$year,results)
lines(ar$year,ar$DAX)

##the latter is almost as good as the optimal

#Exercise
#--------
#Use glmnet to investigate your own data set and find a good value for lambda.

am <- as.data.frame(airmiles)

plot(am)

am$year <- seq_len(nrow(am))
x=matrix(am$year,ncol=1)
x<-cbind(x,I(am$year^2))
x<-cbind(x,I(am$year^3))
x<-cbind(x,I(am$year^4))
x<-cbind(x,I(am$year^5))
x<-cbind(x,I(am$year^6))
x<-cbind(x,I(am$year^7))

##lasso is l1 norm, in glmnet \alpha=1 (Default) for this.
fit7<-glmnet(x,am$x,family="gaussian")
results<-predict(fit7,newx=x,type="response")

plot(am$year,am$x)
for (i in 1:ncol(results)){  
  lines(am$year,results[,i])
}