# RAG System

## This repository contains the RAG development code for the OVPRI AI Assistant project.
Note: BASH scripts are written for running on the HPC server Hickory

### Initial Set Up
To run this project, create a new Python virtual environment and install the dependencies in 'requirements.txt':
>> python -m venv venv_name
>> source venv_name/bin/activate
>> pip install -r requirements.txt

### Pre-Processing
To convert .pdf and .docx documents to .txt, run 'convert_formats.py':
>> python convert_formats.py
Note: Either change the expected path name, or store your source documents under 'data/HRPP'.

To normalize these documents, run 'normalize.py':
>> python normalize.py

To embed these documents into a local Chroma Vector Database, run 'embedding.py':
>> sbatch bash_scripts/embedding.sh

### Launching the RAG system
To run the LLM and launch the front end website, submit the script 'run_llm.sh':
>> sbatch bash_scripts/run_llm.sh

On your local machine's terminal, open a port into Hickory:
>> ssh -N -L 8501:hickory:8501 your-vcu-username@hickory.cs.vcu.edu
Note: This command will appear to hang, this is the expected behavior. Do not close the terminal window.

Access the website at the URL: http://0.0.0.0:8501