import os
from flask import Flask

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.llms.mistralai import MistralAI
from flask import request, render_template
import os
import sys
from typing import List,Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
# llama index imports
import llama_index.core
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,SummaryIndex, StorageContext, Settings, load_index_from_storage, Response
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter, CodeSplitter, LangchainNodeParser
from llama_index.core.tools import FunctionTool,QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters,FilterCondition
from llama_index.core.objects import ObjectIndex
from llama_index.readers.file import IPYNBReader, PandasCSVReader
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
    AgentFnComponent,
    AgentInputComponent
)
from llama_index.readers.file import IPYNBReader, PandasCSVReader
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
    CustomQueryComponent,
    FnComponent
)
from llama_index.core.callbacks import CallbackManager

# llama index agent imports
from llama_index.core.agent import FunctionCallingAgentWorker, ReActAgent, Task, AgentChatResponse, AgentRunner, QueryPipelineAgentWorker

# llama index llms and embeddings imports
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# custom package imports
from llama_index.packs.tables.chain_of_table.base import ChainOfTableQueryEngine, serialize_table

# langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# tools
import nest_asyncio # to allow running async functions in jupyter
import chromadb # persistent storage for vectors
# import nbconvert
import tree_sitter
import tree_sitter_languages
import phoenix as px
from pyvis.network import Network



# global configuration
app = Flask(__name__)
temperture = 0.0 #for deterministic results
llm_model = "mistral-large-latest"
MISTRAL_API_KEY =  "BWdlihu9sUh5P2g3bHnzjAaHiT4anTVH"
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
llm = MistralAI(model=llm_model, temperature=temperture)
agent = None
df = None



def load_data():
    # loading file
    global df
    file_path = "./data_csv/Sepsis_Processed_IC.csv"
    df = pd.read_csv(file_path)
    return df

def setup_qp_table():
    global df
    # create prompt modules
    instruction_str = (
        "1. Convert the query to executable Python code using Pandas.\n"
        "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
        "3. The code should represent a solution to the query.\n"
        "4. PRINT ONLY THE EXPRESSION.\n"
        "5. Do not quote the expression.\n"
    )

    pandas_prompt_str = (
        "You are working with a pandas dataframe in Python.\n"
    
        "The name of the dataframe is `df`. You should interpret the columns of the dataframe as follows: \n 1) Each row represents patient data related to sepsis diagnosis. 2) The Target column indicates whether the patient had sepsis. 3) The duration_since_reg column describes the patient's stay after admission in days. 4) Diagnosis-related columns detail specific diagnostic results and associated codes. 5) The dataset includes patient demographics age, clinical measurements (crp, lacticacid, leucocytes), and diagnostic procedures (diagnosticartastrup, diagnosticblood, etc.). 6) The dataframe also records clinical criteria for sepsis (sirscritheartrate, sirscritleucos, etc.), resource usage, and event transitions (e.g., CRP => ER Triage). 7) Additional columns capture organ dysfunction, hypotension, hypoxia, suspected infection, and treatment details like infusions and oliguria. 8) The dataset covers the transitions between various clinical events, highlighting the pathways in the patient's diagnostic and treatment journey. 9) ER here refers to the emergency room. 10) You only answer questions related to the dataframe. 11) If you do not know the answer, then say you do not know. \n\n"

        "This is the result of `print(df.head())`:\n"
        "{df_str}\n\n"
        
        "Follow these instructions:\n"
        "{instruction_str}\n"
        "Query: {query_str}\n\n"
        "Expression:"
    )
    response_synthesis_prompt_str = (
        "Given an input question, synthesize a response from the query results.\n"
        "Query: {query_str}\n\n"
        "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
        "Pandas Output: {pandas_output}\n\n"
        "Response: "
    )

    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str, df_str=df.head(30)
    )
    pandas_output_parser = PandasInstructionParser(df)
    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

    # define query pipeline with modules
    qp_table = QP(
        modules={
            "input": InputComponent(),
            "pandas_prompt": pandas_prompt,
            "llm1": llm,
            "pandas_output_parser": pandas_output_parser,
            "response_synthesis_prompt": response_synthesis_prompt,
            "llm2": llm,
        },
        verbose=True,
    )

    qp_table.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
    qp_table.add_links(
        [
            Link("input", "response_synthesis_prompt", dest_key="query_str"),
            Link(
                "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
            ),
            Link(
                "pandas_output_parser",
                "response_synthesis_prompt",
                dest_key="pandas_output",
            ),
        ]
    )
    # add link from response synthesis prompt to llm2
    qp_table.add_link("response_synthesis_prompt", "llm2")

    return qp_table

def setup_model_evaluation():
    # evaluate model
    def evaluate_model(model_name:str) -> dict:
        """Load the trained model, evaluation data and evaluate the loaded model."""
        from sklearn.metrics import roc_auc_score, roc_auc_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef
        from joblib import dump, load
        from xgboost import XGBClassifier

        model_save_path = 'models'
        model = load(f'./{model_save_path}/{model_name}.joblib')
        X_test = pd.read_csv('./data_python/Sepsis_X_test.csv')
        y_test = pd.read_csv('./data_python/Sepsis_y_test.csv')

        pred_prob = model.predict_proba(X_test) # get the prediction probabilities for the test set
        predictions = model.predict(X_test) # get the predictions for the test set

        roc_auc = roc_auc_score(y_test, pred_prob[:,1]) # calculate the roc auc score
        average_precision = average_precision_score(y_test, pred_prob[:,1]) # calculate the
        mcc =  matthews_corrcoef(y_test, predictions)
        f1_macro = f1_score(y_test, predictions, average='macro')
        cm = confusion_matrix(y_test, predictions)

        return {"roc_auc":roc_auc, "average_precision":average_precision, "mcc":mcc, "f1_macro":f1_macro, "confusion_matrix":cm}

    # create tools
    evaluate_model_tool = FunctionTool.from_defaults(name="evaluate_model", fn=evaluate_model)
    tools = [evaluate_model_tool]

    # setup ReAct agent
    # model_agent_prompt = """You are a proficient python developer. Respond with the syntactically correct code for the question below. Make sure you follow these rules:
    #                                         1. Use context to understand the APIs and how to use them.
    #                                         2. Ensure all the requirements in the question are met.
    #                                         3. Ensure the output code syntax is correct.
    #                                         4. All required dependencies should be imported above the code.
    #                                         Question:
    #                                         {question}
    #                                         Context:
    #                                         {context}
    #                                         Helpful Response:"""
    # model_agent_prompt = PromptTemplate(model_agent_prompt)
    agent_model = ReActAgent.from_tools(tools=tools, 
                                        llm=llm,
                                        verbose=True)
    # agent_model.update_prompts({"agent_worker:system_prompt": model_agent_prompt})
    return agent_model

def setup_multi_agent():
    # initialize the agents
    agent_model = setup_model_evaluation()
    qp_table = setup_qp_table()

    # methods for running the agents / query pipelines
    def run_agent(query: str) -> str:
        """Run the agent model on the query to get evaluation results from trained model."""
        response = agent_model.query(query)
        return str(response)

    def run_query_pipeline(query: str) -> str:
        """Run the query pipeline to analyze dataset for the given query."""
        response = qp_table.run(
            query_str=query,
        )
        return str(response.message.content)

    # create tools
    run_agent_tool = FunctionTool.from_defaults(name="run_agent", fn=run_agent)
    run_query_pipeline_tool = FunctionTool.from_defaults(name="run_query_pipeline", fn=run_query_pipeline)
    agent_tools = [run_agent_tool, run_query_pipeline_tool]

    top_level_agent_prompt = """
                    You are designed to help with a variety of tasks, from answering questions \
                    to providing summaries to other types of analyses.

                    ## Tools
                    You have access to a wide variety of tools. You are responsible for using
                    the tools in any sequence you deem appropriate to complete the task at hand.
                    This may require breaking the task into subtasks and using different tools
                    to complete each subtask.

                    You have access to the following tools:
                    {tool_desc}

                    ## Output Format
                    To answer the question, please use the following format.

                    ```
                    Thought: I need to use a tool to help me answer the question.
                    Action: tool name (one of {tool_names}) if using a tool.
                    Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
                    ```

                    Please ALWAYS start with a Thought.

                    Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

                    If this format is used, the user will respond in the following format:

                    ```
                    Observation: tool response
                    ```

                    You should keep repeating the above format until you have enough information
                    to answer the question without using any more tools. At that point, you MUST respond
                    in the one of the following two formats:

                    ```
                    Thought: I can answer without using any more tools.
                    Answer: [your answer here]
                    ```

                    ```
                    Thought: I cannot answer the question with the provided tools.
                    Answer: Sorry, I cannot answer your query.
                    ```

                    ## Additional Rules
                    - You MUST obey the function signature of each tool. Do NOT pass in no arguments if the function expects arguments.
                    - For queries that clearly involve data retrieval or manipulation (like 'analyze sales data', 'show trends in data'), use 'run_query_pipeline'.
                    - For queries that directly relate to model performance or evaluation (like 'what is the AUC_ROC score', 'evaluate the prediction accuracy'), use 'run_agent'.

                    ## Current Conversation
                    Below is the current conversation consisting of interleaving human and assistant messages.
                    """
    top_level_agent_prompt = PromptTemplate(top_level_agent_prompt)
    agent = ReActAgent.from_tools(tools=agent_tools, 
                                        llm=llm, 
                                        verbose=True)
    agent.update_prompts({"agent_worker:system_prompt": top_level_agent_prompt})
    return agent

def initial_setup():
    global agent
    load_data()
    agent = setup_multi_agent()


@app.route("/query", methods=["GET"])
def query_index():
    # query_text = request.args.get("text", None)
    # if query_text is None:
    #     return (
    #         "No text found, please include a ?text=blah parameter in the URL",
    #         400,
    #     )
    # response = agent.query(query_text)
    # return str(response), 200

    query_text = request.args.get("text", None)
    if query_text is None:
        return (
            "No text found, please include a ?text=blah parameter in the URL",
            400,
        )
    response = agent.query(query_text)
    return render_template('query.html', query_text=query_text, response=response), 200

@app.route("/initialize", methods=["GET"])
def initialize():
    initial_setup()
    # return "initialized", 200
    return render_template('initialize.html'), 200

@app.route("/")
def home():
    # return "go to /query?text=your_query_here to get a response from the agent. <br> <br> go to /initialize to initialize the agent. <br> <br> <br> example query: <br> what is the auc_roc score of the trained XGBoost model? <br> how many positive cases? <br> what is the average time patients spend in the hospital?"
    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)



# example query
# what is the auc_roc score of the trained XGBoost model?
# how many positive cases?
# what is the average time patients spend in the hospital?