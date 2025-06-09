import functools
import os
from typing import Any, Generator, Literal, Optional
import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from databricks_langchain.genie import GenieAgent
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from pydantic import BaseModel

############################################
# Define your LLM endpoint and system prompt
############################################

# TODO: Replace with your model serving endpoint
# multi-agent Genie works best with claude 3.7 or gpt 4o models.
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)


###################################################
## Create a GenieAgent with access to a restaurant Space
###################################################

# TODO add GENIE_SPACE_ID and a description for this space
# You can find the ID in the URL of the genie room /genie/rooms/<GENIE_SPACE_ID>
RESTAURANTS_GENIE_SPACE_ID = "01f0454f200b1b55af90b008a0f406b6"
restaurants_genie_agent_description = "This genie agent can answer questions using google map data. If this agent doesn't answer the question, try a different agent"

restaurants_genie_agent = GenieAgent(
    genie_space_id=RESTAURANTS_GENIE_SPACE_ID,
    genie_agent_name="Genie",
    description=restaurants_genie_agent_description,
    client=WorkspaceClient(
        host=os.getenv("DB_MODEL_SERVING_HOST_URL"),
        token=os.getenv("DATABRICKS_GENIE_PAT"),
    ),
)

###################################################
## Create a GenieAgent with access to a Hotel Space
###################################################

# TODO add GENIE_SPACE_ID and a description for this space
# You can find the ID in the URL of the genie room /genie/rooms/<GENIE_SPACE_ID>
HOTEL_GENIE_SPACE_ID = "01f04559ea9411eabb95003c453e1414"
hotel_genie_agent_description = "This genie agent can answer questions using Booking.com hotel data. If this agent doesn't answer the question, try a different agent"

hotel_genie_agent = GenieAgent(
    genie_space_id=HOTEL_GENIE_SPACE_ID,
    genie_agent_name="Genie",
    description=hotel_genie_agent_description,
    client=WorkspaceClient(
        host=os.getenv("DB_MODEL_SERVING_HOST_URL"),
        token=os.getenv("DATABRICKS_GENIE_PAT"),
    ),
)

# No additional agents needed - using only hotel and restaurant Genie agents

###################################################
## Create a GenieAgent with access to an Airbnb Space
###################################################

# TODO add GENIE_SPACE_ID and a description for this space
# You can find the ID in the URL of the genie room /genie/rooms/<GENIE_SPACE_ID>
AIRBNB_GENIE_SPACE_ID = "01f0456ab0761ebf9acc5f70f7f32a4d"
airbnb_genie_agent_description = "This genie agent can answer questions using Airbnb rental data. If this agent doesn't answer the question, try a different agent"

airbnb_genie_agent = GenieAgent(
    genie_space_id=AIRBNB_GENIE_SPACE_ID,
    genie_agent_name="Genie",
    description=airbnb_genie_agent_description,
    client=WorkspaceClient(
        host=os.getenv("DB_MODEL_SERVING_HOST_URL"),
        token=os.getenv("DATABRICKS_GENIE_PAT"),
    ),
)

#############################
# Define the supervisor agent
#############################

# TODO update the max number of iterations between supervisor and worker nodes
# before returning to the user
MAX_ITERATIONS = 3

worker_descriptions = {
    "restaurants_genie": restaurants_genie_agent_description,
    "hotel_genie": hotel_genie_agent_description,
    "airbnb_genie": airbnb_genie_agent_description,
}

formatted_descriptions = "\n".join(
    f"- {name}: {desc}" for name, desc in worker_descriptions.items()
)

system_prompt = f"Decide between routing between the following workers or ending the conversation if an answer is provided. \n{formatted_descriptions}"
options = ["FINISH"] + list(worker_descriptions.keys())
FINISH = {"next_node": "FINISH"}

def break_down_question(question):
    breakdown_prompt = """Break down this question into specific subtasks for different agents. Each subtask should focus on
    one specific type of information (Airbnb, restaurant, or hotel data). If the question only needs one type of information,
    return 'SINGLE_TASK'.

    Examples:
    Input: "Tell me some airbnb with accessibility in San francisco. Also provide me recommendation of some restaurants with vegan option"
    Output:
    1. Find Airbnb listings in San Francisco that have accessibility features
    2. Search for restaurants in San Francisco that offer vegan options

    Input: "What are the best Italian restaurants in the city?"
    Output: SINGLE_TASK

    Current question: {question}
    Output:"""

    preprocessor = RunnableLambda(lambda _: [{"role": "user", "content": breakdown_prompt.format(question=question)}])
    breakdown_chain = preprocessor | llm
    result = breakdown_chain.invoke({}).content
    return result.strip()

def route_to_agent(task):
    # Define keywords and patterns for each agent
    route_patterns = {
        "restaurants_genie": ["restaurant", "food", "eat", "dining", "vegan", "vegetarian", "cuisine"],
        "hotel_genie": ["hotel", "stay", "booking", "accommodation", "lodging"],
        "airbnb_genie": ["airbnb", "rental", "accessibility", "accessible", "apartment"]
    }

    task_lower = task.lower()
    scores = {}

    for agent, keywords in route_patterns.items():
        score = sum(1 for keyword in keywords if keyword in task_lower)
        scores[agent] = score

    # Return the agent with the highest score, if any keywords matched
    max_score = max(scores.values())
    if max_score > 0:
        return max(scores.items(), key=lambda x: x[1])[0]
    return None

def route_subtasks(state):
    # Get all subtasks if they exist
    if "subtasks" not in state or not state["subtasks"]:
        return state

    current_subtask = state["subtasks"][state["current_subtask_index"]]
    next_agent = route_to_agent(current_subtask)

    if next_agent:
        return {"next_node": next_agent}
    return {"next_node": "FINISH"}

def supervisor_agent(state):
    count = state.get("iteration_count", 0) + 1
    if count > MAX_ITERATIONS:
        return FINISH

    # Get the current question
    current_question = state["messages"][-1]["content"]

    # Check if this is a new question or a follow-up from a subtask
    if "subtasks" not in state:
        # Break down the question
        breakdown_result = break_down_question(current_question)

        if breakdown_result != "SINGLE_TASK":
            # Store subtasks in state
            subtasks = [task.strip() for task in breakdown_result.split('\n') if task.strip() and not task.startswith('Input:')]
            state["subtasks"] = subtasks
            state["current_subtask_index"] = 0
            state["subtask_results"] = []

            # Route the first subtask
            return route_subtasks(state)
    elif "subtasks" in state:
        # Store the result for the previous subtask if available
        if len(state["messages"]) > 0 and state.get("next_node") != "router":
            state["subtask_results"].append(state["messages"][-1])

        # Move to next subtask if available
        state["current_subtask_index"] += 1
        if state["current_subtask_index"] < len(state["subtasks"]):
            return route_subtasks(state)
        else:
            # All subtasks complete, proceed to final answer
            return FINISH

    # For single tasks, use the original routing logic
    class nextNode(BaseModel):
        next_node: Literal[tuple(options)]

    preprocessor = RunnableLambda(
        lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
    )
    supervisor_chain = preprocessor | llm.with_structured_output(nextNode)
    next_node = supervisor_chain.invoke(state).next_node

    if state.get("next_node") == next_node:
        return FINISH

    return {
        "iteration_count": count,
        "next_node": next_node
    }

#######################################
# Define our multiagent graph structure
#######################################


def extract_urls(text):
    # Simple URL extraction
    words = text.split()
    urls = []
    for word in words:
        if word.startswith(("http://", "https://")):
            # Clean up any punctuation at the end of URLs
            url = word.rstrip('.,!?)')
            urls.append(url)
    return urls

def agent_node(state, agent, name):
    result = agent.invoke(state)
    content = result["messages"][-1].content
    
    # Extract URLs from the content
    urls = extract_urls(content)
    
    return {
        "messages": [
            {
                "role": "assistant",
                "content": content,
                "name": name,
                "urls": urls
            }
        ]
    }


def final_answer(state):
    if "subtask_results" in state and state["subtask_results"]:
        # For multi-task questions, combine results from all subtasks
        prompt = """Synthesize a complete answer from the following subtask results.
        Make sure to include all relevant links and URLs in your response.
        Combine the information coherently to answer the original user question:
        Original Question: {original_question}
        Subtasks and Results:
        {subtask_results}
        """
        # Format subtask results
        original_question = state["messages"][0]["content"]
        subtask_info = []
        all_urls = []
        
        for i, (task, result) in enumerate(zip(state["subtasks"], state["subtask_results"])):
            result_info = [f"Subtask {i+1}: {task}\nResult: {result['content']}"]
            
            # Include URLs if they exist
            if 'urls' in result and result['urls']:
                urls_text = "\nRelevant links: " + ", ".join(result['urls'])
                result_info.append(urls_text)
                all_urls.extend(result['urls'])
            
            subtask_info.append("\n".join(result_info))
            
        formatted_prompt = prompt.format(
            original_question=original_question,
            subtask_results="\n\n".join(subtask_info)
        )
    else:
        # For single-task questions, use simple prompt
        formatted_prompt = "Using only the content in the messages, respond to the previous user question using the answer given by the other assistant messages."

    preprocessor = RunnableLambda(
        lambda state: state["messages"] + [{"role": "user", "content": formatted_prompt}]
    )
    final_answer_chain = preprocessor | llm
    return {"messages": [final_answer_chain.invoke(state)]}


class AgentState(ChatAgentState):
    next_node: str
    iteration_count: int


restaurants_genie_node = functools.partial(agent_node, agent=restaurants_genie_agent, name="restaurants_genie")
hotel_genie_node = functools.partial(agent_node, agent=hotel_genie_agent, name="hotel_genie")
airbnb_genie_node = functools.partial(agent_node, agent=airbnb_genie_agent, name="airbnb_genie")

workflow = StateGraph(AgentState)
workflow.add_node("restaurants_genie", restaurants_genie_node)
workflow.add_node("hotel_genie", hotel_genie_node)
workflow.add_node("airbnb_genie", airbnb_genie_node)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("final_answer", final_answer)

workflow.set_entry_point("supervisor")
# We want our workers to ALWAYS "report back" to the supervisor when done
for worker in worker_descriptions.keys():
    workflow.add_edge(worker, "supervisor")

# Let the supervisor decide which next node to go
workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next_node"],
    {**{k: k for k in worker_descriptions.keys()}, "FINISH": "final_answer"},
)
workflow.add_edge("final_answer", END)

multi_agent = workflow.compile()