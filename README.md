# material-science-expert

This project uses a Retrieval-Augmented Generation (RAG) approach to simulate a chatbot expert in material science. It integrates tools like Qdrant, LangChain, and MatSciBERT for efficient document retrieval and processing, offering a chatbot interface and an API data extraction service for seamless user experience.
The system leverages Ollama, where different nodes specialize in specific LLM tasks based on unique system prompts. Using LangChain's StateGraph, workflows are designed to model the system as a graph, with each node interacting with a specialized LLM. This divide-and-conquer approach ensures that outputs from one node are passed to others, producing coherent and meaningful results.

### Prerequisites

Before running the project, ensure you have the following installed:

- [Docker](https://www.docker.com/)
- Python 3.10.5+ (if running without Docker)

### Installation and Setup

#### Clone the Repository

```bash
git clone https://github.com/saj3sh/material-science-expert.git
cd material-science-expert
```

#### Setting up environment file

Create a file named `.env` with following information:

```env
USE_LOCAL_QDRANT=<True or False>

QDRANT_URL=<Qdrant URL when using Qdrant cloud>

QDRANT_TOKEN=<API key for the Qdrant cloud>

MATERIAL_PROJECT_TOKEN=<API key for fetching docs from Material Project>
```

#### Running with Docker

1. To build and run chatbot interface use following syntax. The UI should be accessible on port 8501 [http://localhost:8501](http://localhost:8501/)
   ```bash
   docker compose up --build chatbot
   ```
2. To build and run API data extractor use following syntax.
   `bash
    docker compose run --build api-data-extractor
    `
   **Warning**: The data extractor script has side effects. For the purpose of this project, the setup is simple - During a new data fetch operation it drops entire collection from the Qdrant store first then re-inserts the documents as they are retrieved. The script also has warnings and requires user confirmation before proceeding

#### Running Without Docker

If you prefer not to use Docker, follow these steps:

1. Install Poetry with pip:
   ```bash
   pip install poetry
   ```
2. Install project dependencies:
   ```bash
   poetry install
   ```
3. To run the **Chatbot**, execute:

```bash
poetry run streamlit run chatbot.py
```

The Chatbot will be available at [http://localhost:8501](http://localhost:8501/). 4. To run the **API Data Extractor**, execute:

```bash
poetry run python api_data_extractor.py
```

### Project Structure

```plaintext
.
├── Dockerfile
├── LICENSE
├── README.md
├── api_data_extractor.py
├── chatbot.py
├── config.py
├── docker-compose.yml
├── poetry.lock
├── pyproject.toml
├── streamlit_components
│   ├── __init__.py
│   ├── page_styles.py
│   └── sidebar.py
└── utils
    ├── __init__.py
    ├── data_formatting.py
    ├── embedding_models.py
    ├── embeddings.py
    ├── prompts.py
    ├── qdrant_client.py
    ├── state_graph.py
    └── vocab_mappings.txt
```

1. `chatbot.py` - interface file for the chatbot based on Stream-lit python framework.
2. `api_data_extractor.py` - python script file to fetch data from the MP API
3. `utils/` - a module of custom utility methods and classes, including tools for data formatting, generating embeddings and tokens, system prompts, class for representing different nodes for system graph, and vocabulary mapping.

### StateGraph design

![alt text](https://github.com/saj3sh/material-science-expert/blob/main/state-graph-design.png?raw=true)

Each of these node functions is defined as member methods of a custom class `state_graph.MatSciStateGraph`. The workflow setup is handled in the `chatbot.py` file, and all system prompts are located in `utils.prompts`.

### Limitations

After many trials and errors, I realized that the Ollama 3.2 1B is relatively small and better suited for basic tasks. Additionally, since we only had access to the online model and were working within a limited timeframe, domain-based fine-tuning wasn't feasible—something that could have significantly improved accuracy. That said, the system design for this task is modular, allowing us to easily switch to a larger model with more parameters such as OpenAI or Anthropic, which should effectively boost prediction accuracy.

### Future Enhancements

Currently, only the vector embeddings are stored in Qdrant, with MP-ID as the sole metadata. I believe we could improve accuracy by integrating an SQL query retrieval system alongside the vector database. This system would generate SQL queries based on user prompts and retrieve relevant data from relational databases when available. Such integration would be particularly useful for performing comparative analysis on different material properties.
