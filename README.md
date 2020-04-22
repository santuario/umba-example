# UMBA - Code example

[TrackStreet](https://www.trackstreet.com) has built an industry leading *SaaS* platform that leverages the power of artificial intelligence and automation to radically reduce brand and pricing policy violations. 

![Trackstreet Web](./images/1.png)

In order to achieve this, a series of web crawlers were set up to download the information of different products featured in a series of online stores (BestBuy, Amazon, Target, ...), and analyze their name, description, price and image. In order to know if a product belonged to one of the clients, in my role as a Data Scientist, I was commissioned to solve a supervised text classification problem, with the goal of working out which supervised machine learning methods were best suited to solve it.

### Problems that were solved

* Finding a way to choose between ML models (different model types, tuning parameters, and features) to *predict* **Product ID** from **Product Name**.
* Applying a **Model Evaluation Procedure** to estimate how well a model will generalize to *out-of-sample* data.
* Choosing and fine tuning a **Model Evaluation Metric** to quantify the model performance.

### Review of the ML Metrics

* **Model evaluation procedure**: K-fold cross-validation
* **Model evaluation metrics**: F1 Score

### Workflow

1. [Reading a text-based dataset](#1-reading-a-text-based-dataset)
2. Choose n-gram or n-char approximation
3. Vectorizing our dataset 
4. Building a model
5. Comparing models
6. Choose a model
7. Identify Outliers
8. Improve hyperparameters
9. Build a class
10. Run a cron job for trainning





## Hands On

### 1. Reading a text-based dataset


This project is simple Lorem ipsum dolor generator.

![Trackstreet Web](./images/2.png)



*trijicon 6 x 48 mm acog Dual iluminación ámbar Chevron .223 retículo) 6,5 Moa RMR 2 Vista óptica ta75 Mount, color negro*

Product Name (*X*) | UPC | Product ID (*Y*)
--- | --- | ---
Trijicon ACOG | 658010111379 | 4
Trijicon ACOG 6x48mm Chevron | 658010111546 |  whats4
Trijicon 6X48 acog. TA648-50G | 658010111249 | 4
Ultimate Flora Women's Care Probiotic 15 Billion | 631257158789 | 52
Renew Life Women's Care Ultimate Flora Probiotic | 658010111379 | 52
Ultimate Flora Probiotic 15 Billion | 658012531373 | 52
Renew Life Everyday Ultimate Flora Probiotic | 631257158772 | 52



	
## Technologies
Project is created with:
* Lorem version: 12.3
* Ipsum version: 2.33
* Ament library version: 999


```python
# Model evaluation metrics: F1 Score
from sklearn.metrics import f1_score


def f1_multilabel(estimador, X, y):
    preds = estimador.predict(X)
    return f1_score(y, preds, average="micro")
```
	
## Setup
To run this project, install it locally using npm:

```
$ cd ../lorem
$ npm install
$ npm start
```

# BONUS


As you can see, I breath and live technology; but all the babble about the next best way of using AI, ML, or Data for the sole purpose of monetizing and implement it around the world absolutely bores me to tears... 


I can be reading a really cool article on *Wired Magazine* about this new fridge that uses AI to predict when you need to buy more sparkling water or something and, for me, it is like watching paint dry. I couldn't care less. What I *do* care about is what you do with it: how do you take this algorithm based technology and make it live in places it's never been before... What I'm really interested in is the convergence of *Data* and *AI/ML* and ethics as a tool for creative troubleshooting. For example: 


**1)** In order to understand the *social mobility* based on skin tone, [I calculated the PANTONE®* of mexican political parties](https://www.youtube.com/watch?v=AZcDNoGPaVg) 

**2)** I used Big Data on *Twitter* in order to see [the age at which sexual assaults begins in Mexico](https://translate.google.com/translate?hl=en&sl=es&tl=en&u=https://verne.elpais.com/verne/2016/05/19/mexico/1463615258_699475.html) 

**3)** Or I made a *LinkedIn*'s web scraping to realize if exists a [*shortcut* to be CEO in Mexico](https://translate.google.com/translate?hl=en&sl=es&tl=en&u=https%3A%2F%2Fwww.unocero.com%2Fredes-sociales%2Fblancos-hombres-de-escuelas-privadas-y-bilingues-asi-son-los-ceos-en-mexico%2F&sandbox=1) 





