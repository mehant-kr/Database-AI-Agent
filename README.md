# Chatbot_With_UI

Simple chatbot with a flask web interface.

The chatbot leverages OpenAI API for user query comprehension complemented with RAG for up-to-date information retrieval.

Tools Used: 
  1. LangChain framework: for user query synthesis and reply generation
  2. Flask: for web gui
  3. GPT-4o mini: for language understanding
  4. Database: Information retreval from the connected database using RAG
     Chinook Database is used: https://www.kaggle.com/datasets/anurag629/chinook-csv-dataset?select=Customer.csv

<br>


### **Steps to run:**

1. Download this github repository
2. Create a virtual environment

     `python -m venv venv`
3. Activate the virtual environment

    `source venv/bin/activate` or `venv/bin/activate` [for Windows]

4. Install the requirements

   `pip install -r requirements.txt`

5. Add your OpenAI API key in a .env file
6. On the terminal run the command below 

     `python app.py`

7. App should not be running on localhost default port

<br>

<p align="center">
<img align="center" width="724" alt="Screen Shot 2024-01-11 at 1 49 47 PM" src="https://github.com/mehant-kr/Database-AI-Agent/blob/c6bb44a1229d110cdf4008ae45bc56abe9b0976d/assets/ss_ai_database_chatbot.png" >
</p>
