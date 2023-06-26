# Alzheimer's Diagnosis

![ADNI Picture](http://adni.loni.usc.edu/wp-content/themes/freshnews-dev-v2/images/ADNI_logo_vector.png "ADNI")

# Table of Contents
1. [Summary of Alzheimer's](https://github.com/amrMohammmed/AD-Project.git#summary-of-alzheimers-disease)
2. [Project Motivation](https://github.com/amrMohammmed/AD-Project.git#project-motivation)
3. [First Data Set](https://github.com/amrMohammmed/AD-Project.git#first-data-set-adni-q3)
4. [Prediction Models, First Data Set](https://github.com/amrMohammmed/AD-Project.git#prediction-models)


## Summary of Alzheimer's Disease
Alzheimer's disease (AD) is a progressive neurodegenerative disease. Though best known for its role in declining memory function, symptoms also include: difficulty thinking and reasoning, making judgements and decisions, and planning and performing familiar tasks. It may also cause alterations in personality and behavior. The cause of AD is not well understood. There is thought to be a significant hereditary component. For example, a variation of the APOE gene, APOE e4, increases risk of Alzheimer's disease. Pathologically, AD is associated with [amyloid beta plaques](https://www.google.com/search?hl=en&authuser=0&biw=1500&bih=687&tbm=isch&sa=1&ei=kf95XM7aDqq4jwS8gLq4Dw&q=amyloid+plaques&oq=amyloid+plaques&gs_l=img.3..35i39j0l7.4609.6096..6209...0.0..0.53.317.7......1....1..gws-wiz-img.GSgXEP-kU3g) and [neurofibrillary tangles](https://www.google.com/search?hl=en&authuser=0&tbm=isch&q=amyloid+plaques&chips=q:amyloid+plaques,g_1:neurofibrillary+tangles:g4CvXoEy7h0%3D&usg=AI4_-kTMw9QaPmdY4wGL4xAlH9TlVhV6-w&sa=X&ved=0ahUKEwjtrcviyOLgAhWI5YMKHSUWCAIQ4lYIKigB&biw=1500&bih=687&dpr=1).

### Diagnosis
Onset of the disease is slow and early symptoms are often dismissed as normal signs of aging. A diagnosis is typically given based on history of illness, cognitive tests, medical imaging, and blood tests.

### Treatment
There is no medication that stops or reverses the progression of AD. There are two types of drugs that attempt to treat the cognitive symptoms:
* Acetylcholinesterase Inhibitors that work to prevent the breakdown of acetylcholine, a neurotransmitter critical in memory and cognition. 
* Memantine (Namenda), which works to inhibit NMDA receptors in the brain.

These medications can slightly slow down the progression of the disease.

### Prevention
It is thought that frequent mental and physical exercise may reduce risk.

---

## Project Motivation
The Alzheimer's Association estimates nearly 6 million Americans suffer from the disease and it is the 6th leading cause of death in the US. The estimated cost of AD was $277 billion in the US in 2018. The association estimates that *early and accurate* diagnoses could save up to $7.9 trillion in medical and care costs over the next few decades. 

Sources: [Mayo Clinic](https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447 "Mayo Clinic - Alzheimer's Disease"), [Alzheimer's Association](https://www.alz.org/alzheimers-dementia/facts-figures), [Wikipedia](https://en.wikipedia.org/wiki/Alzheimer's_disease)


### Project Description: 
Using data provided by the [ADNI Project](http://adni.loni.usc.edu/), it is our goal to develop a computer model that assists in the diagnosis of the disease. We will try multiple models recently popularized in machine learning (Neural Network, SVM, etc.) and more traditional statistical models such as ordinal regression, multinomial regression, and decision trees. 

---
## First Data Set: ADNI Q3
* 628 observations, 15 features (will likely use subset of features)
* Labels: (CN, LMCI, AD)
* Class Label distribution:
![alt-text](https://github.com/amrMohammmed/AD-Project/blob/5b1d1c0a4a8903b453d1f1bb04fa849d3466abaa/Dx.PNG?raw=true "Class Distribution Image")
* Features include age, gender, years of education, race, genotype, cognitive test score (MMSE), and more

* There are six error scenarios:

| Prediction    | Actual        |Error Type        |
| ------------- |:-------------:|:-------------:|
| CN     | LMCI | False Negative |
| CN     | AD      | False Negative |
| LMCI | CN      | False Positive |
| LMCI     | AD | ? |
| AD     | CN      | False Positive |
| AD | LMCI      | ? |

**Important Note:** The models using this data set assume the physician diagnoses (DX.bl) are correct.

---
## Prediction Model

### Multi-Class Prediction in Python 
* Since the data was processed with Scikit-Learn, it was easy to try several models using the library such as logistic regression.

**Results:** Test Accuracy between (79-95)%
