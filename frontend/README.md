🔍 VaultQA — Deep Research AI for Private Documents

VaultQA is a local-first AI research assistant that allows users to upload documents and ask questions grounded strictly in their content.

The system combines Retrieval-Augmented Generation (RAG), hybrid retrieval, and a multi-step Deep Research agent pipeline to produce structured research reports with verified citations.

Unlike traditional chatbots, VaultQA performs:
	•	intent classification
	•	research planning
	•	hybrid document retrieval
	•	evidence reranking
	•	multi-step reasoning
	•	answer verification

before generating responses.

The entire system runs locally using Ollama LLMs, ensuring data privacy and transparency.

🚀 Features

Document Grounded Question Answering

Answers are generated only from uploaded documents using a vector search pipeline.

🧠 Deep Research Mode

Complex questions trigger a multi-step research workflow:
	1.	Query intent classification
	2.	Research plan generation
	3.	Hybrid retrieval (semantic + keyword)
	4.	Evidence reranking
	5.	Sub-question reasoning
	6.	Structured report synthesis
	7.	Evidence verification

⚡ Real-Time Streaming Responses

Both Ask mode and Research mode stream responses to the UI for faster interaction.

🔎 Hybrid Retrieval System

VaultQA retrieves evidence using:
	•	semantic embeddings
	•	keyword search
	•	FAISS vector similarity

📚 Evidence-Backed Answers

Every answer includes:
	•	supporting citations
	•	document page references
	•	extracted evidence snippets

🛡 Verification and Safety

The system performs:
	•	answer verification
	•	contradiction detection
	•	grounding checks

🏗 System Architecture
User Query
     │
     ▼
Intent Classifier
     │
     ▼
Research Planner
     │
     ▼
Hybrid Retriever
(semantic + keyword)
     │
     ▼
FAISS Vector Store
     │
     ▼
Reranker
     │
     ▼
Evidence Context Builder
     │
     ▼
LLM Generation (Ollama)
     │
     ▼
Verification + Contradiction Detection
     │
     ▼
Final Answer / Research Report

🛠 Tech Stack

Backend
	•	Python
	•	FastAPI
	•	FAISS vector database
	•	SentenceTransformers embeddings
	•	Ollama local LLM runtime

🤖 AI Components
	•	Intent classification
	•	Hybrid retrieval
	•	Evidence reranking
	•	Multi-step reasoning
	•	Evidence verification

💻 Frontend
	•	Next.js
	•	TypeScript
	•	TailwindCSS
	•	Framer Motion

📁 Project Structure
vaultqa-rag
│
├── backend
│   └── app
│       ├── main.py
│       ├── config.py
│       ├── middleware.py
│       └── rag
│           ├── engine.py
│           ├── deep_research.py
│           ├── hybrid_retriever.py
│           ├── keyword_retriever.py
│           ├── query_intent.py
│           ├── research_planner.py
│           ├── reranker.py
│           ├── verifier.py
│           ├── contradiction.py
│           ├── chunking.py
│           └── store.py
│
├── frontend
│   └── src
│       ├── app
│       └── lib
│
├── data
│   └── docs
│
└── README.md

⚙ Installation

Clone Repository
git clone https://github.com/YOUR_USERNAME/vaultqa-rag.git
cd vaultqa-rag

🔧 Backend Setup

Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

Install dependencies
pip install fastapi uvicorn pydantic
pip install sentence-transformers
pip install faiss-cpu
pip install numpy
pip install requests
pip install pypdf

🤖 Install Ollama

Download Ollama
https://ollama.ai

Run a model
ollama run qwen2.5:7b

or
ollama run llama3

▶ Start Backend
uvicorn backend.app.main:app --reload --port 8000

API available at
http://127.0.0.1:8000

💻 Frontend Setup
cd frontend
npm install
npm run dev

Frontend runs at
http://localhost:3000

💡 How It Works
1.	Upload documents using the UI
2.	Documents are chunked and indexed into FAISS
3.	Ask questions about the documents
4.	Enable Deep Research mode for complex analysis

VaultQA will:
	•	retrieve relevant document sections
	•	generate reasoning steps
	•	synthesize a research report
	•	display supporting citations

🧪 Example Query
Summarize the key themes of this document and explain their historical context

VaultQA will:
	1.	classify the query intent
	2.	generate sub-questions
	3.	retrieve supporting evidence
	4.	produce findings
	5.	synthesize a structured report

📊 Evaluation

VaultQA includes a simple evaluation dataset:
backend/app/eval/testset.json

Used for measuring:
	•	retrieval accuracy
	•	grounding quality
	•	answer relevance

🔮 Future Improvements
Potential future enhancements:
	•	knowledge graph generation
	•	agentic multi-hop retrieval
	•	reranker model training
	•	optimized streaming inference
	•	evaluation benchmarks
	•	multi-document reasoning

👩‍💻 Author
Sakshi Tapkir
MS Information Systems
Northeastern University
