from langchain.prompts import PromptTemplate

MAX_LIMIT = 10

prompt_generate_related_attributes = PromptTemplate(
    template="""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|> 
    You are a materials science expert with access to a database of materials. Each material has a set of predefined attributes, such as:

    1. **Material ID**
    2. **Whether the material is theoretical** or not
    3. **Structural properties** (e.g., lattice parameters, angles, species present, charge)
    4. **Energy properties** (e.g., energy per atom, formation energy, energy above the hull)
    5. **Stability** (e.g., stable/unstable)
    6. **Electronic properties** (e.g., band gap, whether the band gap is direct or indirect, material type - metallic or non-metallic)
    7. **Magnetic properties** (e.g., total magnetization, normalized magnetization by volume, magnetic site count)
    8. **Mechanical properties** (e.g., bulk modulus (VRH), shear modulus (VRH), universal anisotropy value)
    9. **Surface properties** (e.g., values such as surface energy and work function)
    10. **Possible charged species** (e.g., ions)
    11. **Decomposition information** (e.g., decomposition to other materials, decomposition amounts)

    Try to include as many attributes as possible in the response. **Always include Material ID**

    Based on the human's query, you will identify the relevant material properties that will help answer the question. The response should be a **JSON object** that includes:

    1. **"related_attributes"**: An array of strings representing the relevant attributes identified. Include all relevant attributes that may answer the human's question based on the properties available in the material's database entry.
    2. **"is_context_available"**: Set to **"true"** if at least one attribute was identified as relevant, otherwise set to **"false"**.

    The response should **only contain the key and value**, without any additional explanations or preambles:

    <|eot_id|>
    <|start_header_id|>human<|end_header_id|> 
    {query}  
    <|eot_id|>
    <|start_header_id|>ai<|end_header_id|> 
    """,
    input_variables=["query"],
)

prompt_required_data_points = PromptTemplate(
    template="""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|> 
    You are a material science expert. Determine the minimum number of materials whose information is necessary to answer the human's query. Be as precise and minimal as possible.
    
    Format your response as a JSON object with the following key:
    1. "required_data_points": An integer that represents the minimum number of materials needed to answer the human's query. If the integer value is greater than 10, set it to 10.
    The response should contain only the key and value, with no additional explanations or preambles.
    <|eot_id|>
    <|start_header_id|>human<|end_header_id|>
    {query} 
    <|eot_id|>
    <|start_header_id|>ai<|end_header_id|>
    
    """,
    input_variables=["query"],
)

prompt_summarize_conversation = PromptTemplate(
    template="""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|> 
    You are asked to **only rephrase** the last query made by the human, based on the ongoing conversation, so that it stands alone as a clear and independent question.

    **Do not answer the human's query. Do not provide any details or context.** Simply rephrase the last query so it can stand alone without needing further context.

    Conversation History:
    {chat_history}

    Rephrase the human's last query to make it a clear and independent question.
    <|eot_id|>
    <|start_header_id|>ai<|end_header_id|>

    """,
    input_variables=["chat_history", "query"],
)

# todo - finetune prompt
prompt_generate_search_query = PromptTemplate(
    template="""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|> 
    You are a materials science expert. Your task is to rephrase the human's query to explicitly include provided material properties (attributes). 

    Focus on integrating material properties (attributes) into human's query from the following list: 

    {related_attributes}

    **Avoid asking the human for clarification or additional details under any circumstances.** 

    <|eot_id|>
    <|start_header_id|>human<|end_header_id|> 
    {query}
    <|eot_id|> 
    <|start_header_id|>ai<|end_header_id|> 
    """,
    input_variables=["related_attributes", "query"],
)

# todo - finetune prompt
prompt_generate_final_response = PromptTemplate(
    template="""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|> 
    You are a material science expert. Use the following pieces of context to answer the human's question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    **Try to answer in tables when applicable. Prioritize answering using tables.**
    Contexts:
    {contexts}

    **Reminder**: Always prioritize using tables when answering questions. If you provide any data or properties, structure the response in a table.
    **Reminder**: Don't use tables when answering questions if you cannot answer based on above context.
    <|eot_id|>
    <|start_header_id|>human<|end_header_id|>
    {query}
    <|eot_id|>
    <|start_header_id|>ai<|end_header_id|>
    
    """,
    input_variables=["contexts", "query"],
)

__all__ = [
    "prompt_generate_related_attributes",
    "prompt_required_data_points",
    "prompt_summarize_conversation",
    "prompt_generate_search_query",
    "prompt_generate_final_response"
]
