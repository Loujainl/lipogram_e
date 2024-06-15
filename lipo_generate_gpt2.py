import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class SimpleTextGenerator:
    def __init__(self, model_path):
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    def clean_text(self, text):
        # Remove tokens containing the letter "e"
        return ' '.join(token for token in text.split() if 'e' not in token.lower())

    def generate_text(self, prompt, method='sampling', **generation_kwargs):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


sampling_parameters = {
    'do_sample': True,
    'max_new_tokens': 1000,
    'min_length': 1000,
    'top_k': 50,
    'top_p': 0.45,
    'temperature': 0.5,
    'repetition_penalty': 2.0,
    'num_return_sequences': 1,
    'pad_token_id': GPT2Tokenizer.from_pretrained("gpt2").eos_token_id,
}

# Example usage
model_path = "./fine-tuned-model-clean"  # Path to the fine-tuned model
# prompt = "production d'un discours 1000 mots: "
# prompt = "Dans un but strict, formulons un script sans un certain symbol. production d'un discours 1000 mots: "
prompt = "Bonjour tout la population du pays"  # Initial prompt


text_generator = SimpleTextGenerator(model_path)

# Generate text iteratively until the total word count is at least 1000
total_generated_text = ""
while len(total_generated_text.split()) < 1000:
    generated_text = text_generator.generate_text("C", method='sampling', **sampling_parameters)
    cleaned_text = text_generator.clean_text(generated_text)

    total_generated_text += "\n " + cleaned_text

# Ensure the final text meets the minimum length requirement
if len(total_generated_text.split()) > 1000:
    total_generated_text = ' '.join(total_generated_text.split()[:1000])

print("Generated text with sampling:")
print(total_generated_text)

