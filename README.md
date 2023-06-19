<img src="./extra/Lopouratory.png" height="128"> 

### Citation

Nunez, M. D., Charupanit, K., Sen-Gupta, I., & Lopour, B. A., Lin, J. J. (2022).
[Beyond rates: Time-varying dynamics of high frequency oscillations as a biomarker of the seizure onset zone.](https://iopscience.iop.org/article/10.1088/1741-2552/ac520f/meta) Journal of Neural Engineering. 19(1), 016034.




# sozhfo  
#### (Repository version 0.3.1)

**Authors: Michael D. Nunez, Krit Charupanit, Indranil Sen-Gupta, Beth A. Lopour, and Jack J. Lin from the University of California, Irvine**


### Prerequisites

[MATLAB](https://www.mathworks.com/)

[MCMC Sampling Program: JAGS](http://mcmc-jags.sourceforge.net/)

[Scientific Python libraries](https://www.continuum.io/downloads)

[Python Repository: pyjags](https://github.com/tmiasko/pyjags)


### Downloading

The repository can be cloned with `git clone https://github.com/mdnunez/sozhfo.git`

The repository can also be may download via the _Download zip_ button above.

### Data availability

Automatically identified HFO and qHFO counts per second, standardized delta (1-4 Hz) power, channel localization labels, and samples from posterior distributions for Model 2 and Model 3 are available upon request and on [Figshare](https://doi.org/10.6084/m9.figshare.12385613). Samples from posterior distributions for Model 1 are available upon request.

### Processing Steps

1. hfo_extractHFO.m (Extract HFO candidates)
2. hfo_extractqHFO.m (Find likely artifact HFOs)
3. hfo_fitmodel14.py (Model 1 in Paper)
4. hfo_fitmodel12.py (Model 2 in Paper)
5. hfo_fitmodel8.py (Model 3 in Supplementals)
6. hfo_sleepeval.py (Evaluate relationship of model states to sleep stage and/or delta power)
7. hfo_SOZprediction_avg_nomixture.py (Evaluate prediction of SOZ by HFO rate, CV, and clumping parameters from Model 1)
8. hfo_SOZprediction_avg_nomix_boot.py (Generate ''Strong Prediction'' baseline through label mixing with parameters from Model 1)
9. hfo_SOZprediction_avg.py (Evaluate prediction of SOZ by parameters from Model 2)
10. hfo_SOZprediction_avg_boot.py (Generate ''Strong Prediction'' baseline through label mixing with parameters from Model 2)

### License

sozhfo is licensed under the GNU General Public License v3.0 and written by Michael D. Nunez, Krit Charupanit, Indranil Sen-Gupta, Beth A. Lopour, and Jack J. Lin from the University of California, Irvine.


