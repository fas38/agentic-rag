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
    # prompt modules
    instruction_str = (
        "1. Convert the query to executable Python code using Pandas.\n"
        "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
        "3. The code should represent a solution to the query.\n"
        "4. PRINT ONLY THE EXPRESSION.\n"
        "5. Do not quote the expression.\n"
        "6. Add axis labels, legend, and title when creating a plot.\n"
    )

    pandas_prompt_str = (
        "You are working with a pandas dataframe in Python.\n"
        "The name of the dataframe is `df`. You should interpret the columns of the dataframe as follows: \n"
        "1) Each row represents patient data related to sepsis diagnosis.\n"
        "2) The Target column indicates whether the patient had sepsis.\n"
        "3) The duration_since_reg column describes the patient's stay after admission in days.\n"
        "4) Diagnosis-related columns detail specific diagnostic results and associated codes.\n"
        "5) The dataset includes patient demographics age, clinical measurements (crp, lacticacid, leucocytes), and diagnostic procedures (diagnosticartastrup, diagnosticblood, etc.).\n"
        "6) The dataframe also records clinical criteria for sepsis (sirscritheartrate, sirscritleucos, etc.), resource usage, and event transitions (e.g., CRP => ER Triage).\n"
        "7) Additional columns capture organ dysfunction, hypotension, hypoxia, suspected infection, and treatment details like infusions and oliguria.\n"
        "8) The dataset covers the transitions between various clinical events, highlighting the pathways in the patient's diagnostic and treatment journey.\n"
        "9) ER here refers to the emergency room.\n"
        "10) You only answer questions related to the dataframe.\n"
        "11) If you do not know the answer, then say you do not know.\n\n"

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
    def evaluate_model(model_name:str, threshold:int = -1) -> dict:
        """Load the trained model, evaluation data and evaluate the loaded model."""
        from sklearn.metrics import roc_auc_score, roc_auc_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef
        from joblib import load
        
        model_save_path = 'models'
        model = load(f'./{model_save_path}/{model_name}.joblib')
        X_test = pd.read_csv('./data_python/Sepsis_X_test.csv')
        y_test = pd.read_csv('./data_python/Sepsis_y_test.csv')

        pred_prob = model.predict_proba(X_test) # get the prediction probabilities for the test set
        predictions = model.predict(X_test) # get the predictions for the test set

        roc_auc = roc_auc_score(y_test, pred_prob[:,1]) # calculate the roc auc score
        average_precision = average_precision_score(y_test, pred_prob[:,1]) # calculate the
        best_threshold = model.best_threshold_

        if threshold > 0:
            predictions = np.where(pred_prob[:,1] > threshold, 1, 0)
            mcc = matthews_corrcoef(y_test, predictions)
            f1_macro = f1_score(y_test, predictions, average='macro')
            cm = confusion_matrix(y_test, predictions)

        else:
            mcc =  matthews_corrcoef(y_test, predictions)
            f1_macro = f1_score(y_test, predictions, average='macro')
            cm = confusion_matrix(y_test, predictions)

        return {"roc_auc":roc_auc, "average_precision":average_precision, "mcc":mcc, "f1_macro":f1_macro, "confusion_matrix":cm, "best_threshold":best_threshold}
    
    # uncertainty quantification
    def conformal_prediction(model_name:str, alpha:int = -1) -> dict:
        """Load the trained model, do uncertainty quantification on the loaded model and return the coverage and average width of the prediction sets."""
        from joblib import load
        from crepes import WrapClassifier
        
        """ loading the model and data"""
        model_save_path = 'models'
        model = load(f'./{model_save_path}/{model_name}.joblib')
        X_cal = pd.read_csv('./data_python/Sepsis_X_cal.csv')
        y_cal = pd.read_csv('./data_python/Sepsis_y_cal.csv').to_numpy().reshape(-1)
        X_test = pd.read_csv('./data_python/Sepsis_X_test.csv')
        y_test = pd.read_csv('./data_python/Sepsis_y_test.csv').to_numpy().reshape(-1)

        """calibrating the model"""
        wrapped_clf = WrapClassifier(model) 
        wrapped_clf.calibrate(X_cal, y_cal)
        
        """ uncertainty quantification - coverage and average width of the prediction sets"""
        if alpha > 0:
            prediction_sets = wrapped_clf.predict_set(X_test, confidence=(1-alpha))
            coverage = np.mean([prediction_sets[i][y_test[i]] for i in range(len(prediction_sets))])
            widths = [np.sum(pred) for pred in prediction_sets] 
            average_width = np.mean(widths)
        else:
            alpha = 0.1
            prediction_sets = wrapped_clf.predict_set(X_test, confidence=(1-alpha))
            coverage = np.mean([prediction_sets[i][y_test[i]] for i in range(len(prediction_sets))])
            widths = [np.sum(pred) for pred in prediction_sets] 
            average_width = np.mean(widths)
        
        return {"coverage":coverage, "average_width":average_width}

    # venn abers
    def venn_abers_calibration(model_name:str) -> dict:
        """ Load the trained model, do uncertainty quantification using Venn-Abers calibration and generate the prediction intervals for the test set."""

        # load the trained model
        from venn_abers import VennAbersCalibrator, VennAbers
        from joblib import load
        import matplotlib.pyplot as plt

        model_save_path = 'models'
        plot_save_path = 'static/plots'

        va = VennAbersCalibrator() # initialize the Venn-Abers calibrator

        # load the model and data
        model = load(f'./{model_save_path}/{model_name}.joblib')
        X_cal = pd.read_csv('./data_python/Sepsis_X_cal.csv')
        y_cal = pd.read_csv('./data_python/Sepsis_y_cal.csv').to_numpy().reshape(-1)
        X_test = pd.read_csv('./data_python/Sepsis_X_test.csv')
        y_test = pd.read_csv('./data_python/Sepsis_y_test.csv').to_numpy().reshape(-1)

        # model results
        prediction_prob_cal = model.predict_proba(X_cal)
        prediction_prob_test = model.predict_proba(X_test)

        # get calibrated prediction probabilities and predicted class labels
        p_prime = va.predict_proba(p_cal=prediction_prob_cal, y_cal=y_cal, p_test=prediction_prob_test, p0_p1_output=True) # probability intervals for class 1
        y_pred = np.argmax(va.predict(p_cal=prediction_prob_cal, y_cal=y_cal, p_test=prediction_prob_test), axis=1) # predicted class labels

        # get the prediction probabilities and intervals for class 1
        y_pred_interval_p1 = p_prime[1] # intervals for class 1
        y_pred_p1 = p_prime[0][:, 1] # predicted probability of class 1

        # create dataframe using the prediction probabilities and intervals for class 1
        df = pd.DataFrame({'p0': y_pred_interval_p1[:,0], 'p1': y_pred_interval_p1[:,1], 'p of class_1': y_pred_p1, 'y': y_test})


        # sort the predictions based on the predicted probability of class 1
        sorted_indices = np.argsort(y_pred_p1) # sort the predicted probabilities of class 1
        y_pred_interval_p1 = y_pred_interval_p1[sorted_indices]
        y_pred_p1 = y_pred_p1[sorted_indices]
        y_test_sorted = y_test[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]

        # calculate the lower and upper bounds of the intervals for class 1
        lower_bound = y_pred_p1 - y_pred_interval_p1[:, 0] # calculate lower bound by subtracting the lower interval from the predicted probability of class 1
        upper_bound = y_pred_interval_p1[:, 1] - y_pred_p1 # calculate upper bound by subtracting the predicted probability of class 1 from the upper interval 
        bounds = [lower_bound, upper_bound]

        # plot the predicted probability of class 1 with intervals
        plt.figure(figsize=(8, 5))
        plt.errorbar(np.arange(len(y_pred_p1)), y_pred_p1, yerr=bounds, fmt='o', ecolor='tab:red', capsize=5, label='Predicted Probability of Class 1')
        # plt.scatter(np.arange(len(y_pred_p1)), y_test_sorted, color='tab:blue', label='Actual Label')
        plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
        plt.ylabel('Probability')
        plt.xlabel('Test Sample')
        plt.title('Predicted Probability of Class 1 with intervals')
        plt.savefig(f'./{plot_save_path}/predictionIntervals.png')
        plt.close()

        return_path = f'./{plot_save_path}/predictionIntervals.png'
        return {"interval_plot": "the intervals plot has been saved to the static folder", "plot_path": return_path}

    # create tools
    evaluate_model_tool = FunctionTool.from_defaults(name="evaluate_model", fn=evaluate_model)
    conformal_prediction_tool = FunctionTool.from_defaults(name="conformal_prediction", fn=conformal_prediction)
    venn_abers_calibration_tool = FunctionTool.from_defaults(name="venn_abers_calibration", fn=venn_abers_calibration)
    tools = [conformal_prediction_tool, evaluate_model_tool, venn_abers_calibration_tool]

    # setup ReAct agent
    model_agent_prompt = """
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
                - For queries that require uncertainty quantification (like 'what is the coverage and average width of the prediction sets'), use 'conformal_prediction'.
                - For queries that requires to evaluate the model (like 'what is the f1 score of the model'), use 'evaluate_model'.
                - For queries that require Venn-Abers calibration (like 'generate the prediction intervals for the test set'), use 'venn_abers_calibration'.
                - Answer only the questions asked.

                ## Current Conversation
                Below is the current conversation consisting of interleaving human and assistant messages.
                """
    model_agent_prompt = PromptTemplate(model_agent_prompt)
    agent_model = ReActAgent.from_tools(tools=tools, 
                                        llm=llm,
                                        verbose=True)
    agent_model.update_prompts({"agent_worker:system_prompt": model_agent_prompt})
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
    query_text = request.args.get("text", None)
    if query_text is None:
        return (
            "No text found, please include a ?text=blah parameter in the URL",
            400,
        )
    response = agent.query(query_text)
    response = str(response)
    
    # Extract image path from response
    image_path = None
    if '/plots/' in response:
        start_index = response.find('/plots/')
        end_index = response.find('.png', start_index)
        if end_index == -1:
            end_index = len(response)
        image_path = response[start_index:end_index].strip()
        image_path = image_path + '.png'
        response = "The prediction intervals for the trained model"

    return render_template('query.html', query_text=query_text, response=response, image_path=image_path), 200




@app.route("/initialize", methods=["GET"])
def initialize():
    initial_setup()
    # return "initialized", 200
    return render_template('initialize.html'), 200

@app.route("/")
def home():
    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)



# example query
# what is the auc_roc score of the trained XGBoost model?
# how many positive cases?
# what is the average time patients spend in the hospital?
# what is the average time patients spend in the hospital?
# what is the prediction intervals for XGBoost model?"
# what is the uncertainty quantification of the HGBoost model?