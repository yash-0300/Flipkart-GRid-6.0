from crewai import Crew, Process
from agents import product_finder_agent, product_describer_agent, negotiating_agent, deals_offers_agent, recommender_agent, supervisor_agent
from tasks import product_finder_task, product_describer_task, negotiating_task, deals_offers_task, recommender_task, supervisor_task

import os
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAAF1cFQJy_zDZKIx8NhZuHKBExZ7mHYLM'

# Forming the seller-focused crew with some enhanced configurations
crew = Crew(
  agents=[
    product_finder_agent, 
    product_describer_agent, 
    negotiating_agent, 
    deals_offers_agent, 
    recommender_agent
  ],
  tasks=[
    product_finder_task, 
    product_describer_task, 
    negotiating_task, 
    deals_offers_task, 
    recommender_task, 
    supervisor_task
  ],
  process=Process.hierarchical,  # Optional: Sequential task execution is default
  memory=True,
  embedder={
    "provider": "google",
    "config": {
        "model": 'models/embedding-001',
        "task_type": "retrieval_document",
        "title": "Embeddings for Embedchain"
    }
  },
  cache=True,
  manager_agent=supervisor_agent,
  max_rpm=60,
  share_crew=True
)

# Start the task execution process with enhanced feedback
result = crew.kickoff(
  inputs={
    'domain': 'e-commerce product selling',
    'user_query': 'What are the products of Alisha are present?',
    'csv_path': './flipkart_dataset.csv',
    'target_price': '',
    'past_purchases': 'smartphone, laptop',
    'interests': 'technology, photography',
    'available_deals': 'festive discounts, cashback offers'
  }
)

print(result)
