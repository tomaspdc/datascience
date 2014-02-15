##we are going to do some logistic regression!
library(ggplot2)

x <- read.csv("http://www.ats.ucla.edu/stat/data/binary.csv") 
head(x)
table(x)
xtabs(~admit+rank, data=x) # read this as "counts" against admit and rank
help(xtabs)

##lets try some linear models
lin.fit <- lm(admit ~ ., data=x) ## admittance against every other var
summary(lin.fit)
plot(lin.fit)

##we see intercept is low significance
lin.fit2 <- lm(admit ~ 0 + ., data=x)
summary(lin.fit2)
plot(lin.fit2)

##both are poor fits plots if look at the QQ

#lets make the rank a factor for lr
x$rank <- factor(x$rank)

##remember that this is a special version of General Linear Models 
#with bernoulli (logit is the canonical link for bernoulli/binomial error dist.) 
logit.fit <- glm(admit ~ ., family='binomial', data=x)
summary(logit.fit) ## here we see that all of the parameters are significant

#Recall that the response being modelled is the log(odds) that Y=1. The regression coefficients
#give the change  in log(odds) in the response for a unit change in hte predictor
#variable, keeping all the other variables constant. To get to odds, lets exponentiate
#the betas

coef(logit.fit)
exp(coef(logit.fit))

##here we see the odds of P(Y=1) increase by a factor of 2.2345 for each unit increase
##in gpa. Heres the rest of the data:
#(Intercept)         gre         gpa       rank2       rank3       rank4 
#0.0185001   1.0022670   2.2345448   0.5089310   0.2617923   0.2119375 

# coeffs of indicator (dummy) vars are slightly different...
#for example, the coeff of rank2 represents the change in the log-odds of the output variable that comes going to a rank2 school instead of a rank1 school
summary(x)

##now we are going to generate some test data
#this generates some new data using the existing data to caluclate the means
new.data <- with(x, data.frame(gre=mean(gre), gpa=mean(gpa), rank=factor(1:4)))
new.data$rankprob <- predict(logit.fit, newdata=new.data, type='response')

#as we might expect from the coeffeicients, as the rank of the shools decreases, we
#see a decreased likelihood of acceptance into Yale.(is it Yale?)

#from summary(x) we can see min and max of gre, lets keep everything else constant and look at this.
new.data2 <- with(x, 
            data.frame(gre=rep(seq(from=200, to=800, length.out=100), 4), ##repeat this four times
                      gpa=mean(gpa), 
                      rank=factor(rep(1:4, each=100)))) ## give each of the gre sequences a different rschool rank
new.data2$pred <- predict(logit.fit, newdata=new.data2,type='response') #type=response will expoentiate the output to give the odds,
#otherwise you get the logodds.
ggplot(new.data2, aes(x=gre, y=pred)) + geom_line(aes(colour=rank), size=1) #size releates to thickness of curve

##So this graph really shows how LR is working under the good. We see non linear increase with Gre.
##We also see the non linear *decrease" with a change in rank (not how not equ)

new.data3 <- with(x, data.frame(gpa=rep(seq(from=0, to=4.0, length.out=100), 4),
                                gre=mean(gre), 
                                rank=factor(rep(1:4, each=100))))
new.data3$pred <- predict(logit.fit,newdata=new.data3,type='response')
ggplot(new.data3, aes(x=gpa, y=pred)) + geom_line(aes(colour=rank),size=1) 

##again we see non linear relationship of both the variables here. Odds, as expected
##increases much more rapidly with GPA. We see similar decrease in odds as rank of school decreases.

xtabs(~admit + rank,data=x) # note that of the categoricals,  there are no zero/sparse cells

