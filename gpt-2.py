from transformers import pipeline

# Initialize the text generation pipeline
generator = pipeline('text-generation', model='gpt2-medium')

# Generate text given a prompt
output = generator("what is a computer?,", max_length=300)

# Print the output
for i, generated_text in enumerate(output):
    print(f"Generated Text {i+1}: {generated_text['generated_text']}")
