# RAG System

## This repository contains the RAG development code for the OVPRI AI Assistant project.
Note: BASH scripts are written for running on the HPC server Hickory

### Initial Set Up
To run this project, create a new Python virtual environment and install the dependencies in 'requirements.txt':
> python -m venv venv_name  
> source venv_name/bin/activate  
> pip install -r requirements.txt  

### Pre-Processing
To convert .pdf and .docx documents to .txt, run 'convert_formats.py':  
Note: Either change the expected path name, or store your source documents under 'data/HRPP'.  
> python convert_formats.py

To normalize these documents, run 'normalize.py':  
> python normalize.py

To embed these documents into a local Chroma Vector Database, launch the LLM and run 'embedding.py':  
> sbatch bash_scripts/run_llm.sh  
> sbatch bash_scripts/embedding.sh  

### Launching the RAG system
The model used (Llama-3-8b) requires access to be granted. Request this access on HuggingFace and create an access token.  
Store your HuggingFace access token in a file named '.hf_token'  
To run the LLM and launch the backend, submit the script 'run_llm.sh':  
> sbatch bash_scripts/run_llm.sh

To launch the frontend:
> cd frontend/chatbot
> npm run dev

On your local machine's terminal, open a port into Hickory:  
Note: Do not close the terminal window.  
> ssh -L 5001:localhost:5000 your-vcu-username@hickory.cs.vcu.edu

Access the website at the URL: http://localhost:5173/