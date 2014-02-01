### EXAMPLE 1 - supervisor performance ###

# this dataset shows a set of six numeric survey responses Xi (survey responses) 
# and a dependent variable Y (perceived supervisor quality)
# we want to predict Y from the X's
x <- read.table('http://www.ats.ucla.edu/stat/examples/chp/p054.txt', sep='\t', h=T)

head(x)

# this set of scatterplots gives us an idea of the pairwise relationships 
# present in the dataset
plot(x)

# this linear fit represents the "full model"; eg, the fit with all of the 
# independent variables included
fit <- lm(Y ~ ., data=x)
summary(fit)
# residuals not symmetric? you probably have outliers
# media should be ~0 of course

fit2 <- update(fit, .~. -X5)
# remove feature w/ lowest (abs) t score (t value)
summary(fit2)
# note R-sq decreases slightly, but adj R-sq increases slightly
# --> increasing bias, decreasing variance

fit3 <- update(fit2, .~. -X4)
summary(fit3)

fit4 <- update(fit3, .~. -X2)
summary(fit4)			
# stopping criteria met: all featuers have |t| > 1a
# --> optimal bias-variance pt reached
# --> Residual standard error (RSE) minimized

fit5 <- update(fit4, .~. -X6)	
# note this model is weaker (lower R-sq, higher RSE)
summary(fit5)

fit6 <- update(fit5, .~. -X3)	
# weaker still
summary(fit6)

plot(resid(fit4))			
# what you want to see absence of structure in resid scatterplot ("gaussian white noise")
# --> this plot looks pretty good; also note that resid quartiles look good
# --> if you see some structure there there's something in the model that should be added

qqnorm(resid(fit4))		
# want to see straight diagonal line in resid qqplot
# --> again, looks pretty good
# --> if not diagonal === maybe outliers

### EXAMPLE 2 - cigarette consumption ###

x <- read.table('http://www.ats.ucla.edu/stat/examples/chp/p081.txt', sep='\t', h=T)
head(x)
x$State <- NULL			
# remove state label (non numeric)

fit <- lm(Sales ~ ., data=x)
# full model
summary(fit)
# --> Multiple R-squared:  0.3208,  Adjusted R-squared:  0.2282
# two main possible explanations:
# - we don't have the data (some variable we don't have)
# - some variable creating noise in the system
# also look directly at the Residuals: not symmetric

fit <- lm(Sales ~ 0 + ., data=x)
# pin intercept to 0
# your "min" of the data is 0
# the R-squared are calculated from there (look at the summary)
# R-squared shouldn't improve by removing/pinning the intercept

summary(fit)	
# --> Multiple R-squared:  0.9564,  Adjusted R-squared:  0.9506 
# note weird stats! (high R-sq, low t-scores)
# --> linear regression assumps violated
# --> likely explanation: need more data for prediction
# --> we've overfit this model

fit2 <- update(fit, .~. -HS)
summary(fit2)

fit3 <- update(fit2, .~. -Female)
summary(fit3)
# note t-score of Age jumps (Age becomes much more significant)
# --> make sure you remove only one feature at a time with BE!

plot(resid(fit3))			
# obvious outlier present
# --> this is not what you want to see
# --> there's something systematic

qqnorm(resid(fit3))		
# this does not look good! also resid quartiles are out of wack
# --> conclusion: this dataset doesn't support multiple linear regression very well!
# --> next step: before discarding modeling approach, get more data!

# a lot of problems, very little explanatory power

# In this dataset researchers set out to study the blood pressure (sbp) 
# of individuals in Peru who moved from a high altitude to a lower altitude
setwd('/Users/tomaspica/Documents/DataScience/datascience/raw_classes/day01')
peru <- read.table('peru.dat', h=T)
head(peru)
peru.fit <- lm(formula = sbp ~ ., data=peru)
summary(peru.fit)
# Residual standard error: 10.44
# Multiple R-squared:  0.4998,	Adjusted R-squared:  0.3665  

# What happens to the Residual Standard Error and Adjusted R-Squared score when we 
# remove terms from the regression?

library(MASS)
help(step)
step <- stepAIC(peru.fit, direction="both")
# step will do our previous iterative process to find the best fit on the minimum
# variable number
summary(step)

# Call:
lm(formula = sbp ~ years + weight + height + chin, data = peru) 
# The step function finds a model by dropping terms and aims to maximise adjusted R squared
# Residual standard error: 10 
# cool this dropped slightly

# Multiple R-squared:  0.4792,	Adjusted R-squared:  0.4179
# and Adjusted R-squared is up from the previous model. The adjusted R-square takes
# into account the number of terms we need to produce the regression.
# We want to have as few as possible.