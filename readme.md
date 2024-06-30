Initial implementation of agentic rag for multi documents - [example](./multi_agentic_rag.ipynb) using Mistral LLM.

Implementation of agentic rag for jupyter notebooks - [example](multi_agentic_rag_code.ipynb)

#### Multi Agentic RAG
Implementation of rag using llama Index - [source](multi_agentic_rag_code.ipynb)
    
    - separate query pipeline using panadas dataframe for analyzing csv files
    - separate agent for evaluating trained model
    - top level agent for picking the best agent for the task

To run the code using flask server, run the following command:
    
    - python server.py