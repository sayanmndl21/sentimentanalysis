I am doing a Sentiment Analysis of IMDB movie reviews for term project.

First part, I am separating the dataset (50000) into two - Part A - 40000 and Part B - 10000
Here Part A will be used for step 1 and step 2:
Using Markov Text Generator as language model, I am generating a synthetic dataset separately for positive and negative sentiments (MTG is fit separately on positive and negative reviews).
So I have a real world dataset (part B) and a synthetic dataset.
Now using TF-IDF and Naive Bayes as inference model, I am training and evaluating on the synthetic data (step 2 and step 4).
The same process is repeated but I am training an LSTM model and evaluating it on synthetic data (step 3 and step 4).
Step 5:
I will repeat the above processes on Part B which will be the real world dataset i.e training both models on 70% of Part B dataset and evaluating on 30% (step 2,3 and 5).
Results are comparison of all four models.
I would appreciate any feedback on the above, thanks :slightly_smiling_face:
11:26
Optionally, I can evaluate the models trained by synthetic data on the 30% part B data

Done:
1. Synthetic Dataset
2. Train LSTM on Synthetic Dataset (70%) and test on Synthetic Data (30%) and Real World Data(30%)

To Do:
1. Train LSTM on Realworld Data and test on Synthetic Data (30%) and Real World Data(30%) - Sayan
2. Train TF-IDF+Naive Bayes on Synthetic Dataset (70%) and test on Synthetic Data (30%) and Real World Data(30%) - Arjun
3. Train TF-IDF+Naive Bayes on Real World Dataset (70%) and test on Synthetic Data (30%) and Real World Data(30%) - Arjun
(Try to generate confusion matrix and ROC curves along with accuracy scores and examples)
(try to maintain the same vocab, embedding size and min/max length)

4. Prepare outline for document.