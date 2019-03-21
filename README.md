# How is the interest in eco-friendly products evolving within Amazon?

## Abstract

The dataset we have choosen is the Amazon reviews which contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014. This year’s theme is “data science for social good”, we though about how we could improve society through data analysis.

What we want to do : Study and try to tell a story about the evolution of people interest concerning climate change, renewables energies, pollution and bio food. Indeed there are many categories of products to work on : [Books, Kindle Store, TV-Movies, Home and Kitchen, Health - Personal Care, Tools and Home Improvement, Grocery and Gourmet Food]. We will then do some visualization to show the evolution about the concerned products with the Number of Reviews, the Sale Ranking of the product and the number of products linked with our subject.

We would like then to use our Machine Learning knowledge to use the extracted data in order to train some models which could give some predictions about the coming years. This could give some ideas and trends about the products linked to the subject, by showing maybe a growing (or not) market, and an evolution of the interest by the users.

Actually, aside Ada course, our group is following another course Data Visualization in which our project is to do some data visualization about different climates changes scenario over the year 2050. The data is provided by Stanford University and what we would like to do is to compare Amazon and Standford data over two differents project but focused on the same topic.


The main code is : __project_general.pynb__
It is using some python files, that we implemented and created using the different notebooks associated with the categories. 
Indeed we specified the analysis on some particular categories, and we finally merged it. 


## Research questions
- How the interest and concern in the ecological environment evolved over the years ?
- How Amazon proposals evolved on this theme ? Did they follow the trend on this subject
- What are the most concerned categories about environment ? (Garden, Books, Grocery etc...)
- Who are the main and different  brands ? 
- What are the most growing sectors focusing on Ecology ?
- How is evolving the price of these products ? (The bio is for example known to be expensive).
- How could evolve the Sale Ranking of this kind of products ? 
- How could evolve the Nb of reviews from ecofriendly products ? 
- How could evolve the Nb of ecofriedndly products  ?
- How are the ratings of ecofriendly products compared to the general ones ? 

## Dataset
List the dataset(s) you want to use, and some ideas on how do you expect to get, manage, process and enrich it/them. Show us you've read the docs and some examples, and you've a clear idea on what to expect. Discuss data size and format if relevant.

The total amount of data represents 142.8 million reviews spanning from May 1996 to July 2014. It is in total around 20Gb. This dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs).

Here is the link of Amazon dataset overview : http://jmcauley.ucsd.edu/data/amazon/

As our theme is mostly climate change, we might focus on some categories such as :  Books, Kindle Store, TV-Movies, Health - Personal Care, Tools and Home Improvement, App for android, Amazon Instant Video. Indeed some ohters such as sport and outdoor or clothes might not be a big study part unless people interests can be linked. it will then depend on the results.

The data is split into K-core and Ratings only subsets and group by categories.
K-cores (i.e., dense subsets): These data have been reduced to extract the k-core, such that each of the remaining users and items have k reviews each.
Ratings only: These datasets include no metadata or reviews, but only (user,item,rating,timestamp) tuples. Thus they are suitable for use with mymedialite (or similar) packages.

We will also use the metadata dataset which is complementary to the reviews and will give information about the product itself : Sale ranking, price etc...

The data can by read with python, as a python dictionnary object, which is fine for us because we will use python. It also can be written in json file to use another langage. On the dataset overview, they provide a way to read the data into a pandas data frame.
Regarding the size of the data and the subsete, the question we have to answer is if we need the use of apache spark or not ? As the data is split in subset, it will mainly depend on the size of subsets or on what we want to do and how. Also amazon splits the data into file which can be subset as wee saw above, soem file with or without duplicates reviews, with or without the ratings. We might focus on categories subsets which interest us.

For our side, the dataset is stored on the EPFL cluster.

## A list of internal milestones up until project milestone 2

Milestone 2 : 25th November

First of all we will properly define the relevant categories by observing the data and the numbers.
Then we will work on the data extraction of the chosen categories and for each category we will select the products respecting the criterias of our subject (Bio, Ecology, Renewable etc..).
For this we will have to set a list of key words to focus on, in order to filter the data.
There will be of course a big part of data cleaning in order to gather all the relevant informations.
Additionally, as the dataset is on the cluster, we will do some observation extraction of data in order to understand and use the frameworks through the cluster.

We will then work following these milestones :

11th November :
Analysis of the Amazon dataset. Understanding of the cluster process.
Data analysis - extraction : Choice of categories
List of key words to do the fliter search of products.

18th November :
Analysis of the 2 datasets : Amazon Reviews and Metadata, in order to merge/combine these 2. We will then have the information about the products (categories, sales etc..) and the associated reviews.
Data extraction, data wrangling on the interesting parts.
This will then allow us to work for the next milestone and get all the concerned data.

25th November :
Create a final notebook with the explanations about the data collection and cleaning.
Merged dataset with Dataset Reviews and Metadata.
Different cases of data following the categories of products. Each will have a data cleaning.

Then the dataset will be well processed and ready for the Milestone 3.
Indeed, once the data is well gathered and cleaned, we will then be able to work on it using Machine Learning models to do some prediction and to do visualization on it. With the splitted data we will then compare the different categories.


### Data collection and preprocessing

<br>

To simplify what is presented in the notebook, we have decided to perform all these steps for each category in **separate pyton scripts**.

<br>

For each category, we will proceed as follows:
- Import the meta data and reviews that have been downloaded as `json` files
- Extract the relevant and writtable features of meta data
- Save it as a parquet file of the form **`\"category\"_datacleaned`**
- We then do the same with the reviews and save a parquet file of the form **`"category\"_reviews`**
- Next, we **filter the data we have to only keep articles related to ecology/bio/renewable etc...**
- Finally, we join the metadata and review dataset using the product ID, and create a finale parquet file of the form **`\"category\"_review_product_join`**

<br>

Therefore, all you need to do now is execute all the scripts to automatically generate all the parquet files that will be used in tis notebook.

>Note that once these files have been created, we can skip this part to gain a lot of time.

<br>

---
### Understanding the data

In our work, we started by first analyzing each product category available very widely to evaluate the data in each category and to determine which ones will be the most valuable for our goal.

We quickly realized that many categories had no real interest in revealing the importance of environmentally friendly products.

At that point, we decided to study the following categories:
- Grocery and Gourmet Food
- Healthcare
- Patio Lawn and Garden
- Books

For each of these parts, we then extracted all the eco-friendly items. This data was then totally in line with what we were looking for and really allowed us to retrieve interesting information on the interest shown in these products, and above all an evolution over time.

In fact, one part that was key to successfully revealing what we were looking for was in filtering our data. Indeed, if the filtering was not strict enough, we would get too many items, many of which did not correspond to what we were looking for. 
We also realized that the number of eco-friendly items is not very large in some categories, and that if we want to carry out a consistent study and therefore a strict filter, we are forced not to treat certain categories where the number of items would be way too small to have a good analysis.


<br>

---
### Data analysis

<br>

Once the parquet files are generated, we can use them to carry out different analyses, and each time apply it to all the categories studied to compare them with each other.

<br>

**1)** First, we will compute and visualize **the proportion of environmentally friendly products in each category**. This will allow us to quickly realize the importance of these products in a category. 

This analysis is also important because we have large differences in quantities between the different categories, which could influence the procaine analyses.

<br>

**2)** Then, we will extract the publication dates of each article in each category to **plot the evolution of the number of eco-friendly products**. With these curves, we will then really be able to observe the evolution of the green trend over time.

Of course, we can a priori suspect that we will obtain increasing curves, but it is interesting to evaluate this growth and compare it from one product category to another.

<br>

**3)** Another analysis that we will carry out focuses on the comments left on each of the products. We will indeed track **the distribution of the number of comments by product**. In this way, we can assess people's interest in environmentally friendly products and compare this interest across categories.

<br>


**4)** Then, we will look at the **average prices of the products in a category, and their evolution over time**. In this way, we will be able to observe the price difference between eco-friendly and other products, and to observe the evolution of this difference over time.


### Next steps : 

We confirm that the data is well suited for our project, as we are looking for eco-friendly products. 
We succeed to manage large amount of data (some GB per category, and especially 18GB only for the books reviews)

We will for the milestone 3 work on a data story about our case. Indeed we will show properly the trends in the Amazon products for each category about the ecofriendly products. This will to some quantitative and qualitative analysis results. We can already see for example, that the eco-friendly products mostly appeared in 2002/2003 compared to the other products, and the number is growing.
We describe the different steps that we would like to fulfill, in the project_general notebook, in the TO-do list.

### Milestone 3 Update :
Kevin : Preliminary data analysis, problem formulation, coming up with the algorithm, running tests, writing up the report, will work on the final presentation

Eliott : Problem formulation, coding up the algorithm, writing up the report, running tests, will work on the final presentation

Maxence : Problem formulation, coding up the algorithm, running tests, writing up the report, will work on the final presentation

