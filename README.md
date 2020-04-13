# Umba - Code example

[TrackStreet](https://www.trackstreet.com) has built an industry leading SaaS platform that leverages the power of artificial intelligence and automation to radically reduce brand and pricing policy violations. 

![Trackstreet Web](./images/1.png)

For this purpose, a series of web crawlers downloaded the information of different products in a series of online stores (BestBuy, Amazon, Target, ...) to analyze their price, description and image. As a Data Scientist, to know if a product belonged to one of the clients, I had a supervised text classification problem, and my goal was to investigate which supervised machine learning methods was best suited to solve it.

### Objective

* Need a way to choose between models (different model types, tuning parameters, and features).
* Use a **model evaluation procedure** to estimate how well a model will generalize to *out-of-sample* data.
* Requires a **model evaluation metric** to quantify the model performance.

### Review of our model evaluation

* **Model evaluation procedure**: K-fold cross-validation
* **Model evaluation metrics**: F1 Score

### Agenda

1. Reading a text-based dataset
2. Vectorizing our dataset
3. Choose n-gram or n-char approximation
4. Building a model
5. Comparing models
6. Choose a model
7. Improve hyperparameters
8. Build a class
0. Run a cron job for trainning



## Data example
This project is simple Lorem ipsum dolor generator.

![Trackstreet Web](./images/2.png)


Product Text (*X*) | UPC | Product ID (*Y*)
--- | --- | ---
Trijicon ACOG | 658010111379 | 4
Trijicon ACOG 6x48mm Chevron | 658010111546 | 4
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
	
## Setup
To run this project, install it locally using npm:

```
$ cd ../lorem
$ npm install
$ npm start
```

# Bonus

