Bayesian Inference 2
================

This notes brings the important preliminaries for Bayesian Inference.
---------------------------------------------------------------------

-   Knowing the role and meaning of the parameters in a theoretical distribution
-   Computation of Summaries of a distribution, particularly percentiles
-   Aspects of closed form integration (involving univariate functions)
-   Monte Carlo integration
-   Illustrating Simulation functions available in R and compariosn with other appraoches
    -   Simple distributions like Binomial, Beta, Uniform are used.
-   Bayes Formula in discrete case

Functional forms and effect of location / shape parameters<br>
==============================================================

This part helps to visulize the shape and the effect of shifting origin

``` r
x1=runif(100,2,5)
x=seq(from=-2,to=2,by=0.01) #sort(x1,decreasing = FALSE)
#power / exponential
plot(x,x^2,type="l",col="red")
lines(x,2^x)

par(mfrow=c(1,2))
plot(x,x^2,type="l",col="red")
lines(x,2^x)
plot(x,exp(-x),type="l",col="blue")

par(mfrow=c(2,2))
plot(x,x^2,type="l",col="red")
lines(x,2^x)
plot(x,exp(-x),type="l",col="blue")
plot(x,x^2*exp(-x),type="l",col="green")
plot(x,x^2*exp(-x^2),type="l",col="violet")
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-1-1.png)

``` r
#Location shifting
par(mfrow=c(2,2))
plot(x,x^2,type="l",col="red")
lines(x,(x-2)^2)

plot(x,exp(-x^2),type="l",col="red")
lines(x,exp(-(x-0.5)^2))

plot(x,x^2*exp(-x^2),type="l",col="violet")
lines(x,x^2*exp(-(x-1)^2))

#Shape
par(mfrow=c(2,2))
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-1-2.png)

``` r
plot(x,x^2,type="l")
plot(x,x^3,type="l")
plot(x,1/x,type="l")
plot(x,1/x^3,type="l")
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-1-3.png)

``` r
par(mfrow=c(2,2))
plot(x,exp(-x),type="l")
plot(x,exp(-2*x),type="l")
plot(x,exp(-3*x),type="l")
plot(x,exp(-x/2),type="l")
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-1-4.png)

Finding percentiles *p*\[*X* ≤ *x*<sub>*α*</sub>\]=*α* -Example beta distribution

``` r
require(pracma)
```

    ## Loading required package: pracma

``` r
a=2
b=5
f1<-function(q) {(1/beta(a,b))*q^(a-1)*(1-q)^(b-1)}
aa=seq(0,1,by=0.0001)
k1=length(aa)
s=0
for(i in 1:k1)
{
  s[i]=integral(f1,0,aa[i])
}
qu=0.025      #Required percentile
ep=0.0000001 #Required Precision
qq=max(which(s-qu < ep))
aa[qq] #req_percentile
```

    ## [1] 0.0432

``` r
curve(f1)
abline(v=aa[qq],col="red")
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-2-1.png)

Summary of a RV - mean, variance, maximum, minimum,percentiles...
=================================================================

E\[g(x)\] = integral of g(x) and f(x) wrto x over the range of x<br> If g(x)=x we get mean, if g(x)=x^2 and V(x)=E\[X^2\]-E\[x\]^2 is variance, Here f(x) is a pdf of the RV <br>

``` r
require(pracma)
a1=0.5 # a and b should be positive constants
b1=0.5
f1<-function(y) {y*(1/beta(a1,b1))*y^(a1-1)*(1-y)^(b1-1)}
f2<-function(y) {y^2*(1/beta(a1,b1))*y^(a1-1)*(1-y)^(b1-1)}
mo1=integral(f1,0,1)
mo2=integral(f2,0,1)
va=mo2-mo1^2
cbind(mo1,va)
```

    ##      mo1    va
    ## [1,] 0.5 0.125

Percentiles quantiles are cutpoints dividing the range of a probability \#distribution into contiguous intervals with equal probabilities, or dividing the observations in a sample in the same way. <br> There is one \#less quantile than the number of groups created.Percentiles are quantiles that divide a distribution into 100 equal parts<br>

``` r
a2=5
b2=6
p_df<-function(q) {(1/beta(a2,b2))*q^(a2-1)*(1-q)^(b2-1)}
aa=seq(0,1,by=0.0001)
k1=length(aa)
s=0
for(i in 1:k1)
{
  s[i]=integral(p_df,0,aa[i])
}
qu=0.25             #Required percentile
ep=0.0000001    #Required Precision
qq=max(which(s-qu < ep))
aa[qq] #req_percentile
```

    ## [1] 0.3506

``` r
curve(p_df)
abline(v=aa[qq])
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-4-1.png)

Bayes formula for discrete case<br>
===================================

``` r
theta=c(0.1,0.25,0.5,0.75,0.9,0.99)                  #Parameter values
pri_the=c(0.2, 0.1,0.3,0.2,0.1,0.1)                  #Priors
x=7                                     #No of success
n=7                                    #No of trials
marg=pri_the*dbinom(x,n,theta)          #marginal distribution
marg_sum=sum(pri_the*dbinom(x,n,theta))
theta_U=theta     #Req thetas to find posteriors- type theta for all priors or 0.25...
k=which(theta %in% theta_U)
pos_the=marg[k]/marg_sum
cbind(theta=theta[k],prior=pri_the[k],posterior=round(pos_the,5))
```

    ##      theta prior posterior
    ## [1,]  0.10   0.2   0.00000
    ## [2,]  0.25   0.1   0.00004
    ## [3,]  0.50   0.3   0.01378
    ## [4,]  0.75   0.2   0.15696
    ## [5,]  0.90   0.1   0.28121
    ## [6,]  0.99   0.1   0.54801

For general discrete events<br>
===============================

PROBLEM:<br> The chances of 10 employees becoming managers of a certain company are 0.06751937, 0.02554829, 0.10617217, 0.21419476, 0.130974, 0.04465979, 0.04332306, 0.11943548, 0.17432549, 0.07384759. The probabilities that each pass a screening test 0.3, 0.5, 0.8, 0.3, 0.5, 0.4, 0.3, 0.25, 0.45, 0.3 respectively.<br>

``` r
theta=c(1, 2, 3,4,5,6,7,8,9,10)                        #Parameter values
#pri_the=c(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1) 
pri_the=c(0.06751937,   0.02554829, 0.10617217, 0.21419476, 0.130974,   0.04465979, 0.04332306, 0.11943548, 0.17432549, 0.07384759)       #Priors
#Data
p1=0.3                           #p(A/E1)
p2=0.5                            #p(A/E2)
p3=0.8                            #p(A/E3)
p4=0.3                           #p(A/E4)
p5=0.5                            #p(A/E5)
p6=0.4                            #p(A/E6)
p7=0.3                           #p(A/E7)
p8=0.25                            #p(A/E8)
p9=0.45                            #p(A/E9)
p10=0.3                            #p(A/E10)

d_p=c(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10)                  #data_probabilities_Likelihood                               
marg_NR=pri_the*d_p           #marginal distribution
marg_sum=sum(pri_the*d_p)
pos_the=marg_NR/marg_sum

cbind(theta=theta,prior=pri_the,posterior=pos_the)
```

    ##       theta      prior  posterior
    ##  [1,]     1 0.06751937 0.04952115
    ##  [2,]     2 0.02554829 0.03123006
    ##  [3,]     3 0.10617217 0.20765468
    ##  [4,]     4 0.21419476 0.15709818
    ##  [5,]     5 0.13097400 0.16010177
    ##  [6,]     6 0.04465979 0.04367347
    ##  [7,]     7 0.04332306 0.03177470
    ##  [8,]     8 0.11943548 0.07299858
    ##  [9,]     9 0.17432549 0.19178492
    ## [10,]    10 0.07384759 0.05416249

``` r
plot(pri_the,pos_the,cex=1.5)
text(pri_the,pos_the,theta, cex=1, pos=4, col="red")
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-6-1.png)

Generating discrete probabilities
=================================

``` r
theta=c(1, 2, 3,4,5,6,7,8,9,10)                        #Parameter values
k=length(theta)
pri_the1=runif(k,0,1)
pri_the= pri_the1/sum(pri_the1)      #Priors
#Data  #data_probabilities_Likelihood   
pd=0
for(i in 1:k)
{
  pd[i]=runif(1,0,1)
}
#marginal distribution
marg_NR=pri_the*pd           
marg_sum=sum(pri_the*pd)
pos_the=marg_NR/marg_sum
#Results
cbind(theta=theta,prior=pri_the,posterior=pos_the)
```

    ##       theta       prior   posterior
    ##  [1,]     1 0.003249606 0.003056894
    ##  [2,]     2 0.088741346 0.130733562
    ##  [3,]     3 0.126748265 0.144826059
    ##  [4,]     4 0.178445302 0.235003738
    ##  [5,]     5 0.012474701 0.015105222
    ##  [6,]     6 0.119622003 0.131187935
    ##  [7,]     7 0.087215376 0.035264439
    ##  [8,]     8 0.028585072 0.008073052
    ##  [9,]     9 0.191667529 0.173095170
    ## [10,]    10 0.163250799 0.123653928

``` r
plot(pri_the,pos_the,cex=1.5)
text(pri_the,pos_the,theta, cex=1, pos=4, col="red")
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-7-1.png)

Maximum/Minimum
===============

``` r
a=0.5
b=0.5
p_df<-function(y) {(1/beta(a,b))*y^(a-1)*(1-y)^(b-1)}
mi=optimize(p_df,c(0,1))
mx=optimize(p_df,c(0,1),maximum = TRUE)
minimum=mi$objective
maximum=mx$objective
curve(p_df,col="Green")
abline(v=mx$maximum,col="blue")
abline(v=mi$minimum,lty=2,col="Red")
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-8-1.png)

``` r
c(mi$minimum,minimum,mx$maximum,maximum)
```

    ## [1]  0.5000000  0.6366198  0.9999339 39.1508220

``` r
p_df(0.3)
```

    ## [1] 0.6946091

``` r
dbeta(0.3,a,b)
```

    ## [1] 0.6946091

``` r
integral(p_df,0,0.3)  
```

    ## [1] 0.3690101

``` r
pbeta(0.3,a,b)
```

    ## [1] 0.3690101

MC integration<br>
==================

This section illustrates Monte Carlo integration and compares with actual integration. The integrand is $\\frac {x^2}{(x^3+1)}$ First the number of simulation is 10

``` r
k=10 #No of simulation
fn<-function(x){x^2/(x^3+1)} #function to be integrated
LL=0              #Limits
UL=2
ri=runif(k,LL,UL) #Random generator
ri_s=sort(ri,decreasing=FALSE) #For graphing
ri1=fn(ri_s)      #Evaluation
par(mfrow=c(1,2))
plot(ri1,col="red")
curve(fn,from=LL, to=UL)
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-9-1.png)

``` r
require(pracma)
integral(fn,LL,UL)    #actual integration
```

    ## [1] 0.7324082

``` r
(UL-LL)*sum(ri1)/k    #MC answer
```

    ## [1] 0.7994647

Simulation is increased to 100 then to 1000<br><br>
===================================================

The integrand is *s**i**n*<sup>2</sup>*x*
=========================================

``` r
k=100
fn<-function(x){sin(x)^2}
LL=0
UL=pi
ri=runif(k,LL,UL)
ri_s=sort(ri,decreasing=FALSE)
ri1=fn(ri_s)
par(mfrow=c(1,2))
plot(ri1,col="red")
curve(fn,from=LL, to=UL)
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-10-1.png)

``` r
require(pracma)
integral(fn,LL,UL)
```

    ## [1] 1.570796

``` r
(UL-LL)*sum(ri1)/k
```

    ## [1] 1.53033

The integrand is *e*<sup>−2*x*</sup><br>
========================================

``` r
k=1000
fn<-function(x){exp(-2*x)}
LL=0
UL=10
ri=runif(k,LL,UL)
ri_s=sort(ri,decreasing=FALSE)
ri1=fn(ri_s)
par(mfrow=c(1,2))
plot(ri1,col="red")
curve(fn,from=LL, to=UL)
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-11-1.png)

``` r
require(pracma)
integral(fn,LL,UL)
```

    ## [1] 0.5

``` r
(UL-LL)*sum(ri1)/k
```

    ## [1] 0.4917671

Integration using Beta distribution
===================================

``` r
k=10000 
a=2
b=3
fn<-function(y){(1/beta(a,b))*y^(a-1)*(1-y)^(b-1)}
#fn<-function(x){dbeta(x,a,b)}
LL=0.06758599
UL=0.8058796
ri=runif(k,LL,UL)
ri_s=sort(ri,decreasing=FALSE)
ri1=fn(ri_s)
par(mfrow=c(1,2))
plot(ri1,col="red")
curve(fn,from=LL, to=UL,lwd=2,col="blue")
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-12-1.png)

``` r
require(pracma)
integral(fn,LL,UL)
```

    ## [1] 0.95

``` r
(UL-LL)*sum(ri1)/k
```

    ## [1] 0.9522284

``` r
#Check with
pbeta(UL,a,b)-pbeta(LL,a,b)
```

    ## [1] 0.95

Monte Carlo Simulation - U(0,1) behaviour<br>
=============================================

Generating different samples of varied sizes repeated 10 times<br> Mean and variance are calculated for subsamples - 10 in each case

``` r
x1=matrix(runif(50,0,1),5,10) #n - size - 50
x2=matrix(runif(100,0,1),10,10) #n - size - 100
x3=matrix(runif(500,0,1),50,10) #n - size - 500
x4=matrix(runif(2000,0,1),200,10)  #n - size - 2000
m1=0
for(i in 1:10)
{
m1[i]=mean(x1[,i])
}
m2=0
for(i in 1:10)
{
  m2[i]=mean(x2[,i])
}
m3=0
for(i in 1:10)
{
  m3[i]=mean(x3[,i])
}
m4=0
for(i in 1:10)
{
  m4[i]=mean(x4[,i])
}

v1=0
for(i in 1:10)
{
  v1[i]=var(x1[,i])
}
v2=0
for(i in 1:10)
{
  v2[i]=var(x2[,i])
}
v3=0
for(i in 1:10)
{
  v3[i]=var(x3[,i])
}
v4=0
for(i in 1:10)
{
  v4[i]=var(x4[,i])
}
```

Distribution of original data and mean and variance of subsamples of sizes 5, 10, 50, 200
=========================================================================================

``` r
par(mfrow=c(2,2))
boxplot(x1,col=2,main="n50")
boxplot(x2,col=3,main="n100")
boxplot(x3,col=4,main="n500")
boxplot(x4,col=5,main="n2000")
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-14-1.png)

``` r
cbind(m1,m2,m3,m4)
```

    ##              m1        m2        m3        m4
    ##  [1,] 0.5339012 0.6045523 0.5350305 0.5187756
    ##  [2,] 0.7123538 0.6779876 0.4458111 0.5018127
    ##  [3,] 0.4057052 0.5535471 0.4536863 0.4671981
    ##  [4,] 0.4286829 0.5250260 0.5643062 0.4792202
    ##  [5,] 0.3101504 0.3617634 0.5666429 0.4733394
    ##  [6,] 0.5433335 0.3537526 0.5547909 0.4715506
    ##  [7,] 0.5290041 0.5116239 0.5410718 0.5060126
    ##  [8,] 0.5105593 0.3477354 0.5013019 0.4879361
    ##  [9,] 0.5501828 0.4419852 0.5497345 0.4854368
    ## [10,] 0.7106299 0.3443381 0.4592095 0.4885729

``` r
cbind(v1,v2,v3,v4)
```

    ##               v1         v2         v3         v4
    ##  [1,] 0.09714159 0.10344611 0.09761602 0.08702327
    ##  [2,] 0.01709954 0.08120774 0.07729308 0.08150997
    ##  [3,] 0.09004729 0.07875533 0.09468408 0.09119239
    ##  [4,] 0.07964702 0.04908137 0.08293715 0.08210935
    ##  [5,] 0.04543272 0.10983979 0.05244413 0.08009072
    ##  [6,] 0.15389178 0.09096752 0.08403473 0.08289949
    ##  [7,] 0.07784670 0.10536604 0.06084237 0.07467906
    ##  [8,] 0.18521805 0.09653218 0.08353767 0.08008186
    ##  [9,] 0.17453320 0.06410880 0.09870690 0.08176163
    ## [10,] 0.05368495 0.12188061 0.09316967 0.08336888

Distribution of sample means - 10 samples in each size and Mean of means
========================================================================

``` r
par(mfrow=c(2,2))
boxplot(m1,col=2,main="n50")
boxplot(m2,col=3,main="n100")
boxplot(m3,col=4,main="n500")
boxplot(m4,col=5,main="n2000")
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-15-1.png)

``` r
cbind(n50=mean(m1),n100=mean(m2),n500=mean(m3),n2000=mean(m4))
```

    ##            n50      n100      n500     n2000
    ## [1,] 0.5234503 0.4722312 0.5171586 0.4879855

Generating from a discrete distribution<br>
===========================================

``` r
x=c(0:5) #Discrete data
px=c(0.1, 0.2, 0.3, 0.3,0.05,0.05) #respective probabilities
  #dbinom(x,5,0.5)
cpx=cumsum(px) #distribution function
pr=0    #generated variable
ns=10            #No of simulation
for(i in 1:ns)
{
un=runif(1,0,1)
pr[i]= min(which(un<cpx))-1 
}
hist(pr)
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-16-1.png)

``` r
mean(pr)
```

    ## [1] 2.5

``` r
var(pr)
```

    ## [1] 1.611111

``` r
# check with
sum(x*px)
```

    ## [1] 2.15

``` r
sum(x^2*px)-sum(x*px)^2
```

    ## [1] 1.5275

Generating from a binomial distribution (BD) <br>
=================================================

checking with expectation formulas, simulated distribution in r and closed form formula for moments of BD

``` r
n=30 #binomial parameter - no of trials
p_g=0.05 #binomial parameter - proportion of success
x=c(0:n) #its range
px= dbinom(x,n,p_g) #respective probabilities
cpx=cumsum(px) #distribution function
# or cpx=pbinom(x,n,p_g)
pr=0    #generated variable
ns=100            #No of simulation
for(i in 1:ns)
{
  un=runif(1,0,1)
  pr[i]= min(which(un<cpx))-1 
}
hist(pr)
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-17-1.png)

``` r
sim_u=cbind(mean(pr),var(pr)) #User simulation
# check with expectation formulas
formu=cbind(sum(x*px),sum(x^2*px)-sum(x*px)^2)
#or check with simulated distribution in r
sim_r=cbind(mean(rbinom(1000,n,p_g)),var(rbinom(1000,n,p_g)))
#or check with closed form formula for moments of BD
cld=cbind(n*p_g,n*p_g*(1-p_g))
res=rbind(sim_u,formu,sim_r,cld)
row.names(res)=c("sim_u","formula","sim_r","ClosedForm")
colnames(res)=c("mean","var")
res
```

    ##             mean      var
    ## sim_u      1.320 1.330909
    ## formula    1.500 1.425000
    ## sim_r      1.546 1.527203
    ## ClosedForm 1.500 1.425000

Inverse CDF or Inverse transform method - Generating from arbitrary distribution
================================================================================

``` r
require(pracma)
f1<-function(y) {(2/3)*(y+1)} #Range of the RV is LL and UL
LL = 0
UL = 1
aa=seq(LL,UL,by=0.0001) 
k1=length(aa)
s=0
for(i in 1:k1)
{
  s[i]=integral(f1,0,aa[i])
}
rs=0
ns=10
for(j in 1:ns)
{
  qu=runif(1,0,1)      #Required percentile
  ep=0.0000001 #Required Precision
  qq=max(which(s-qu < ep))
  rs[j]=aa[qq] #req_percentile
}
hist(rs)
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-18-1.png)

``` r
sim_u=cbind(mean(rs),var(rs)) #User simulation
# check with expectation formulas
fm1= function(y) {y*(y+1)*(2/3)}
fm2= function(y) {y^2*(2/3)*(y+1)}
mo1=integral(fm1,LL,UL)
mo2=integral(fm2,LL,UL)
va=mo2-mo1^2
formu=cbind(mo1,va)
res=rbind(sim_u,formu)
row.names(res)=c("sim_u","formula")
colnames(res)=c("mean","var")
res
```

    ##              mean        var
    ## sim_u   0.5226500 0.10736576
    ## formula 0.5555556 0.08024691

Inverse CDF or Inverse transform method - Generating from arbitrary distribution - Triangle<br>
===============================================================================================

``` r
require(pracma)
f1<-function(y) {y} #Range of the RV is LL and UL
LL1 = 0
UL1 = 1
f2<-function(y) {2-y} 
LL2 = 1
UL2 = 2
aa=seq(LL1,UL2,by=0.001) 
k1=length(aa)
s=0
for(i in 1:k1)
{
  if(aa[i]<1)
  {
    s[i]=integral(f1,0,aa[i])
  } else
  {
    s[i]=integral(f1,0,UL1)+integral(f2,1,aa[i])
  }
}
rs=0
ns=1000
for(j in 1:ns)
{
  qu=runif(1,0,1)      #Required percentile
  ep=0.0000001 #Required Precision
  qq=max(which(s-qu < ep))
  rs[j]=aa[qq] #req_percentile
}
sim_u=cbind(mean(rs),var(rs)) #User simulation
# check with expectation formulas
fm11= function(y) {y*f1(y)}
fm12= function(y) {y*f2(y)}
fm21= function(y) {y^2*f1(y)}
fm22= function(y) {y^2*f2(y)}
mo1=integral(fm11,LL1,UL1)+integral(fm12,LL2,UL2)
mo2=integral(fm21,LL1,UL1)+integral(fm22,LL2,UL2)
va=mo2-mo1^2
formu=cbind(mo1,va)
res=rbind(sim_u,formu)
row.names(res)=c("sim_u","formula")
colnames(res)=c("mean","var")
res
```

    ##             mean       var
    ## sim_u   1.011354 0.1748600
    ## formula 1.000000 0.1666667

``` r
par(mfrow=c(1,2))
curve(f1,0,1,xlim=c(LL1,UL2))
curve(f2,1,2,add=TRUE)
hist(rs)
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-19-1.png)

Inverse CDF or Inverse transform method - Generating from beta distribution<br>
===============================================================================

``` r
require(pracma)
a=2
b=5
f1<-function(q) {(1/beta(a,b))*q^(a-1)*(1-q)^(b-1)}
aa=seq(0,1,by=0.00001)
k1=length(aa)
s=0
for(i in 1:k1)
{
  s[i]=integral(f1,0,aa[i])
}
rs=0
ns=1000
for(j in 1:ns)
{
qu=runif(1,0,1)      #Required percentile
ep=0.0000001 #Required Precision
qq=max(which(s-qu < ep))
rs[j]=aa[qq] #req_percentile
}
hist(rs)
```

![](BayesianInference2_files/figure-markdown_github/unnamed-chunk-20-1.png)

``` r
sim_u=cbind(mean(rs),var(rs)) #User simulation
# check with expectation formulas
fm1= function(y) {y*(1/beta(a,b))*y^(a-1)*(1-y)^(b-1)}
fm2= function(y) {y^2*(1/beta(a,b))*y^(a-1)*(1-y)^(b-1)}
mo1=integral(fm1,0,1)
mo2=integral(fm2,0,1)
va=mo2-mo1^2
formu=cbind(mo1,va)
#or check with simulated distribution in r
sim_r=cbind(mean(rbeta(1000,a,b)),var(rbeta(1000,a,b)))
#or check with closed form formula for moments of BD
cld=cbind(a/(a+b),a*b/((a+b)^2*(a+b+1)))
res=rbind(sim_u,formu,sim_r,cld)
row.names(res)=c("sim_u","formula","sim_r","ClosedForm")
colnames(res)=c("mean","var")
res
```

    ##                 mean        var
    ## sim_u      0.2856826 0.02516939
    ## formula    0.2857143 0.02551020
    ## sim_r      0.2791726 0.02498679
    ## ClosedForm 0.2857143 0.02551020
