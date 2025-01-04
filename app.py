# app.py

from flask import Flask, request, render_template
from agent import initialize_agent
import os
from tools import run_script
from langchain.agents import initialize_agent as langchain_initialize_agent
from langchain.agents import AgentType, Tool

app = Flask(__name__)

# Initialize agent and llm
try:
    qa, llm = initialize_agent()
    # Define Tools
    tools = [
        Tool(
            name='RunScript',
            func=run_script,
            description='Run a specified Python script and return the output.'
        )
    ]
    # Initialize LangChain Agent with Tools
    agent = langchain_initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
except Exception as e:
    agent = None
    print(f"Error initializing agent with tools: {e}")

@app.route('/', methods=['GET', 'POST'])
def home():
    response = None
    query = None
    if request.method == 'POST':
        query = request.form['query']
        if 'RunScript' in query and agent:
            # Extract script path from the query
            script_path = query.split('RunScript')[-1].strip()
            if os.path.exists(os.path.join('scripts', script_path)):
                response = run_script(os.path.join('scripts', script_path))
            else:
                response = "Script not found."
        elif agent:
            # Use agent to process the query
            response = agent.run(query)
        else:
            # Fallback to qa if agent is not initialized
            response = qa(query)['result']
    return render_template('index.html', query=query, response=response)

if __name__ == '__main__':
    app.run(debug=True)
