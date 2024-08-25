from crewai import Task
from tools import csv_tool, search_tool
from agents import product_finder_agent, product_describer_agent, negotiating_agent, deals_offers_agent, recommender_agent, supervisor_agent

# Task for Product Finder Agent
product_finder_task = Task(
  description=("""
Task: You are a Product Finder Agent specializing in retrieving product information from a CSV file. Your task is to identify the best product options that match the customer's query based on specific attributes such as price, brand, and features.

Instructions:

1. **Load Data:** Use the CSV tool to load the product data.
2. **Search for Match:** Analyze the customer query to identify relevant product attributes.
3. **Filter Products:** Filter the products in the CSV file based on these attributes.
4. **Rank Products:** Rank the filtered products by relevance to the query, considering factors like price, popularity, and customer ratings.
5. **Output:** Provide a list of top product matches along with a brief description of each product's key attributes.

"""),
  expected_output='A list of top product matches based on the query, with key attributes highlighted.',
  tools=[csv_tool],
  agent=product_finder_agent,
)

# Task for Product Describer Agent
product_describer_task = Task(
  description=("""
Task: You are a Product Describer Agent specializing in providing detailed product descriptions. Your task is to gather comprehensive product information using the search tool and summarize it for the customer.

Instructions:

1. **Search Online:** Use the search tool to gather detailed product information from various online sources.
2. **Analyze Information:** Identify key features, specifications, and unique selling points of the product.
3. **Summarize Description:** Summarize the product information in a way that highlights the most important details.
4. **Output:** Provide a concise, informative product description that helps the customer understand the product's value.

"""),
  expected_output='A detailed and informative product description based on the search results.',
  tools=[search_tool],
  agent=product_describer_agent,
)

# Task for Negotiating Agent
negotiating_task = Task(
  description=("""
Task: You are a Negotiating Agent responsible for negotiating the price of a product with the buyer. Your task is to engage in a simulated negotiation that balances the seller's profit margins with the buyer's budget constraints.

Instructions:

1. **Understand Buyer Preferences:** Analyze the buyer's query to understand their budget and preferred price range.
2. **Propose Initial Offer:** Suggest an initial price that aligns with the seller's profit margins while being attractive to the buyer.
3. **Handle Counteroffers:** If the buyer counters, adjust the offer considering both parties' needs.
4. **Finalize the Deal:** Aim to reach a mutually beneficial price agreement. If the buyer's budget is significantly lower, explore additional offers or discounts.
5. **Output:** Provide the final negotiated price and a brief summary of the negotiation process.

"""),
  expected_output='The final negotiated price along with a summary of the negotiation.',
  agent=negotiating_agent,
)

# Task for Deals and Offers Suggesting Agent
deals_offers_task = Task(
  description=("""
Task: You are a Deals and Offers Suggesting Agent responsible for recommending current deals and promotions to the buyer. 
Your task is to identify the most relevant offers based on the buyer's interest and the products they are considering.

Instructions:

1. **Identify Relevant Offers:** Use the product data and buyer's query to find applicable deals and promotions.
2. **Cross-reference Products:** Ensure that the suggested deals are relevant to the products the buyer is interested in.
3. **Present Offers:** Clearly present the deals and highlight the benefits, such as discounts or bundled offers.
4. **Output:** Provide a list of the most attractive deals and promotions that match the buyer's interests.

"""),
  expected_output='A list of relevant deals and promotions tailored to the buyer’s interests.',
  tools=[csv_tool],
  agent=deals_offers_agent,
)

# Task for Recommender Agent
recommender_task = Task(
  description=("""
Task: You are a Recommender Agent specialized in suggesting products based on the buyer’s interests and past purchases. 
Your task is to analyze the buyer's history and preferences to recommend products they are likely to appreciate.

Instructions:

1. **Analyze Buyer History:** Use the CSV tool to analyze past purchases and preferences.
2. **Identify Patterns:** Identify patterns in the buyer’s interests, such as preferred brands, categories, or features.
3. **Generate Recommendations:** Suggest products that align with the buyer’s preferences and past behavior.
4. **Output:** Provide a list of personalized product recommendations, highlighting why each product is a good fit.

"""),
  expected_output='A personalized list of product recommendations based on buyer’s interests and past purchases.',
  tools=[csv_tool],
  agent=recommender_agent,
)

# Task for Supervisor Agent
supervisor_task = Task(
  description=("""
Task: You are the Supervisor Agent responsible for overseeing all interactions and ensuring that the buyer’s query is directed to the appropriate agent. Your role is to manage the workflow, provide feedback, and ensure the final response is accurate and polite.

Instructions:

1. **Monitor Queries:** Analyze the buyer’s query to determine which agent (Product Finder, Describer, Negotiator, Deals, Recommender) should handle it.
2. **Direct Workflow:** Route the query to the most appropriate agent based on the content.
3. **Review Outputs:** Review the outputs of all agents to ensure accuracy, relevance, and politeness.
4. **Final Response:** Combine the outputs from various agents into a cohesive and polite response to the buyer.
5. **Output:** Provide a final response to the buyer, ensuring all their needs are addressed in a clear and professional manner.

"""),
  expected_output='A final, comprehensive response to the buyer, integrating the inputs from all relevant agents.',
  agent=supervisor_agent,
)
