from crewai import Agent
from tools import csv_tool, search_tool
import pathlib
import textwrap
from dotenv import load_dotenv
load_dotenv()

import os

# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")


# Initialize the LLM with Gemini models
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
# call the gemini models
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                           verbose=True,
                           temperature=0.8,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))


# Product Finder Agent
product_finder_agent = Agent(
    role='Product Finder Agent',
    goal='Assist the buyer in finding the product they are looking for by searching the product database.',
    verbose=True,
    backstory=(
        """You are an expert in finding products that match the customer’s needs."
        "Use the product CSV database to find the best matches.
        """
    ),
    tools=[csv_tool],
    llm = llm,
    allow_delegation=False
)

# Product Describer Agent
product_describer_agent = Agent(
    role='Product Describer Agent',
    goal='Provide detailed descriptions of products by searching the web for additional information.',
    verbose=True,
    backstory=(
        "You are a knowledgeable seller who can provide detailed and accurate descriptions of products using web search."
    ),
    tools=[search_tool],
    llm = llm,
    allow_delegation=False
)

# Negotiating Agent
negotiating_agent = Agent(
    role='Negotiating Agent',
    goal='Negotiate the price of products with the buyer to reach a mutually agreeable price.',
    verbose=True,
    backstory=(
        "You are skilled in negotiating prices with buyers to find the best possible deal while ensuring customer satisfaction."
    ),
    tools=[],
    llm = llm,
    allow_delegation=False
)

# Deals and Offers Suggesting Agent
deals_offers_agent = Agent(
    role='Deals and Offers Suggesting Agent',
    goal='Suggest the best deals and offers available to the buyer based on current promotions.',
    verbose=True,
    backstory=(
        "You specialize in identifying and suggesting the best deals and offers available to buyers."
    ),
    tools=[],
    llm = llm,
    allow_delegation=False
)

# Recommender Agent
recommender_agent = Agent(
    role='Recommender Agent',
    goal='Recommend products to the buyer based on their interests and past purchases.',
    verbose=True,
    backstory=(
        "You are an expert in recommending products based on buyer’s interests and past purchases to enhance their shopping experience."
    ),
    tools=[],
    llm = llm,
    allow_delegation=False
)

# Supervisor Agent
supervisor_agent = Agent(
    role='Supervisor Agent',
    goal='Supervise and coordinate the tasks of all seller agents to provide the best possible response to the buyer.',
    verbose=True,
    backstory=(
        """You are the Supervisor Agent responsible for overseeing the workflow of all other seller agents. 
        Your role is to direct user input to the most appropriate agent and ensure the final response is polite and helpful."""
    ),
    tools=[],
    llm = llm,
    allow_delegation=True
)

# Example of using the Supervisor Agent
# user_query = "I'm looking for a good deal on a laptop with 16GB RAM. Can you help?"

# response = supervisor_agent.run(input_text=user_query)

# print(response)
