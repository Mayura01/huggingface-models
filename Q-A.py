from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

# Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'why he made you?',
    'context': 'my name is IntelliAi. i am a artificial intelligent model diveloped by Mayur. for his final year college project.'
}
res = nlp(QA_input)

print(res)

# Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)