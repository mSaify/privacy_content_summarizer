# Privacy Content Summarizer

reference paper - http://ceur-ws.org/Vol-2645/paper3.pdf
presentation - cs19mds11002_privacy_summarization.pptx for details

## Data collection (creating training and test sets)
#### 1. score matcher
     i. use ToSDr api https://tosdr.org/api to get quoted text with label for privacy services
     ii. use only approved cases
     iii. based on similarity score>80%, match quoted text from a to quoted text of publicly available dataset used in paper
     iv. generate a partial training dataset with matched texts.

#### 2. extract actual text from services
    i. from training set services, download the actual html text from privacy pages.
    ii. keep some of the services policy aside for test held_out_test_data
   

#### 3. create source list
    i. use the downloaded input from step 2 and partial training set from step 1
    ii. compare and augment more neutral sentences from the actual source.
    iii. create final labeled training dataset.

#### 4. smote to increase riskier labeled sentences.


## Model Training

#### 1. Use torch Tensor dataLoader to batch test and training dataset.
#### 2. Model training steps
       
       
       a. use word2Vec embeddings to generate vectors for word token
       b. create sentence matrix as explained in paper
       c. neural network architecture
            CONV1D -> Max Pool -> Concat -> Dropout
            
#### 3. Run training 
    
     a. run 20 epochs to train the model in batches
     b. evaluate precision, recall,f1 score
        
      
## Inference / Summary Generation

    a. Summary Extractor has both ways as explained in paper
        Risk Focused Content Extraction
        Risk Coverage Extraction
    
    hyper parameters
        alpha - risk threshold.
        compression ratio - used to calcuate budget.
        


