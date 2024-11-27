from langchain.prompts import PromptTemplate

MAX_LIMIT = 10

prompt_generate_related_attributes = PromptTemplate(
    template="""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|> 
    You are a materials science expert with access to a database of materials. Each material has a set of predefined attributes, such as:

    1. **Material ID** (unique identifier starting with 'mp-' followed by integers)
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
    You are a materials science expert with access to a database of materials. Each material has a set of predefined attributes, such as:

    1. **Material ID** (unique identifier starting with 'mp-' followed by integers)
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

prompt_answer_based_on_context = PromptTemplate(
    template="""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|> 
    You are a material science expert. Use the following context to answer the user's question.

    Context:
    {contexts}

    **Reminder**:
    - **Use material IDs as rows** (e.g., mp-1234, mp-5678) and **properties as columns** (e.g., density, tensile strength) whenever applicable.
    - **Include units in the column headers** (e.g., "Density (g/cm³)", "Tensile Strength (MPa)"), and **ensure numeric values** in the cells whenever possible.
    - If presenting multiple materials, **group similar properties together in one table**.
    - **Ensure consistency**: Each column should contain only **one type of data** (either entirely numeric values with the same unit, or entirely string-based values). Do not mix numeric and string data in the same column.
    - **Only use information provided in the context**; do not infer or generate data.

    **Example format**:
    If showing material properties:
    | Material ID  | Density (g/cm³) | Tensile Strength (MPa) | Melting Point (°C) | Thermal Conductivity (W/m·K) |
    |--------------|-----------------|------------------------|--------------------|----------------------------|
    | mp-1234      | 4.43            | 900                    | 1668               | 21.9                       |
    | mp-5678      | 2.70            | 250                    | 660                |                            |

    **Ensure that each column contains only one data type** (either numeric or string) and that units are in the header. **Don't mix data types within a column.**

    **Only use information provided in the context**. If the context doesn't provide enough details to answer the question, indicate that directly.

    <|eot_id|>
    <|start_header_id|>human<|end_header_id|> 
    {query}
    <|eot_id|> 
    <|start_header_id|>ai<|end_header_id|> 
    """,
    input_variables=["contexts", "query"],
)


prompt_request_refinement = PromptTemplate(
    template="""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|> 
    You are a material science expert. The user has asked a question, but you do not have sufficient context to provide a complete answer based on the given information. 

    Your task is to apologize and request clarification. You have access to a material database with specific fields that can help answer the question. These fields include:

    1. **Material ID** (unique identifier starting with 'mp-' followed by integers)
    2. **Theoretical status** (whether the material is theoretical or not)
    3. **Structural properties** (e.g., lattice parameters, angles, species, charge)
    4. **Energy properties** (e.g., energy per atom, formation energy, energy above the hull)
    5. **Stability** (e.g., stable/unstable)
    6. **Electronic properties** (e.g., band gap, material type)
    7. **Magnetic properties** (e.g., magnetization, magnetic site count)
    8. **Mechanical properties** (e.g., bulk modulus, shear modulus)
    9. **Surface properties** (e.g., surface energy, work function)
    10. **Possible charged species** (e.g., ions)
    11. **Decomposition information** (e.g., decomposition to other materials)

    Ask the user to provide a **Material ID** or other relevant attributes to refine their query. The Material ID should be prioritized when retrieving data.

    **Reminder**: Avoid making up information. Politely inform the user that more information is needed to proceed with their request.

    <|eot_id|>
    <|start_header_id|>human<|end_header_id|> 
    {query}
    <|eot_id|> 
    <|start_header_id|>ai<|end_header_id|> 
    I'm sorry, I don't have enough information to answer your question. Could you please provide more details, such as the **Material ID** (e.g., mp-12345) or any other specific attributes you're looking for? This will help me better understand and answer your query.
    <|eot_id|> 
    """,
    input_variables=["query"],
)


__all__ = [
    "prompt_generate_related_attributes",
    "prompt_required_data_points",
    "prompt_summarize_conversation",
    "prompt_generate_search_query",
    "prompt_answer_based_on_context",
    "prompt_request_refinement"
]
