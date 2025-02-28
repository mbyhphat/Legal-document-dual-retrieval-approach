<div align="center">
    <h1>Sentence-Level Chunking Analysis in Dual-Retrieval</h1>
</div>

## **Introduction**
In this repository, I will try to analyze the pros and cons of sentence-level chunking in a dual-retrieval approach for legal document retrieval.  

**Key contributions:**
* Applied bi-encoder to generate embeddings for each sentence, then calculated document embeddings via mean pooling. 
* Refined rankings with a cross-encoder by aggregating sentence scores through mean pooling.

