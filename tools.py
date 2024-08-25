from crewai_tools import YoutubeChannelSearchTool
import cohere
import langchain
import langchain_core
import langchain_experimental
import pandas as pd
from langchain.agents import Tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.utilities import PythonREPL
from dotenv import load_dotenv
from dotenv import load_dotenv

load_dotenv()


## CSV Agent-----------------------------------------------------------------------------
from crewai_tools import CSVSearchTool

# Initialize the tool with a specific CSV file. This setup allows the agent to only search the given CSV file.
# csv_tool = CSVSearchTool(csv='./flipkart_dataset.csv',
#                          config={
#                         "llm": {
#                         "provider": "google",
#                         "config": {
#                         "model": "gemini-pro",
#                         # "temperature": 0.7,
#                         },
#                         },
#                         "embedder": {
#                         "provider": "google",
#                         "config": {
#                         "model": "models/embedding-001",
#                         "task_type": "retrieval_document",
#                         },
#                         },
#                         }
                         
#                         )

import os
os.environ['GOOGLE_API_KEY'] = 'Paste_your_api_key'

csv_tool = CSVSearchTool(
    csv = './flipkart_dataset.csv',
    config = dict(
        llm = dict(
            provider = "google",
            config = dict(
                model = "gemini-pro",
                temperature = 0.7,
            )
        ),
        embedder = dict(
            provider = "google",
            config = dict(
                model = "models/embedding-001",
                task_type = "retrieval_document",
            )
        )
    )
)



# Google Search api ===============================================================================================================


from crewai_tools import SerperDevTool

# Initialize the tool for internet searching capabilities
search_tool = SerperDevTool()

## Python Agent---------------------------------------------------------------------------------------##

python_repl = PythonREPL()
python_tool = Tool(
    name="python_repl",
    description="Executes python code and returns the result. The code runs in a static sandbox without interactive mode, so print output or save output to a file.",
    func=python_repl.run,
)
python_tool.name = "python_interpreter"

class ToolInput(BaseModel):
    code: str = Field(description="Python code to execute.")
python_tool.args_schema = ToolInput

def run_python_code(code: str) -> dict:
    """
    Function to run given python code
    """
    input_code = ToolInput(code=code)
    return {'python_answer': python_tool.func(input_code.code)}

functions_map = {
    "run_python_code": run_python_code,
}

tools = [
    {
        "name": "run_python_code",
        "description": "given a python code, runs it",
        "parameter_definitions": {
            "code": {
                "description": "executable python code",
                "type": "str",
                "required": True
            }
        }
    },
    ]
