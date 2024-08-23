### Problem Statement:
Information Extraction Web Application
Develop a web application for extracting specific information from sources such as web URLs, PDFs, or text documents.
The application should chunk the extracted content, convert it into vector embeddings using an embedding model of your choice, and store it in an in-memory vector database like FAISS.
When a user provides a conversational query (e.g. use all the facts mentioned in the content and write an article),
the application should retrieve the relevant data chunks, make an API call to the LLM to extract the required information, and display the results in a user-friendly interface.

## Steps for processing the application
1. Have to create function of browsing pdf,txt or web url
2. extract the pdf text to chunks
3. then convert the chunks into vector embedding
   ## Vector Embedding
   A vector embedding is a way to convert text data into a numerical format
    that captures the meaning of the text in a high-dimensional space.
4. Use FAISS(Faebook AI Similarity Search) for easy retrival and search the content
5. provide a structured prompt for model
