#FROM python:3-alpine3.15
FROM python:3.13-slim
WORKDIR /Database-AI-Agent
COPY . /Database-AI-Agent
RUN pip install -r requirements.txt
EXPOSE 3000
# ENV OPENAI_API_KEY=''         # add your API key here
CMD python ./app.py