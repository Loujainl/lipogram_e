import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Step 1: Prepare Your Training Data
train_file = "la_disparation-.txt"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=64  # Adjust block size as needed
)

# Step 2: Load Pre-Trained Model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 3: Fine-Tune the Model
training_args = TrainingArguments(
    output_dir="./fine-tuned-model-clean",
    overwrite_output_dir=True,
    num_train_epochs=5,  # Increase number of epochs
    per_device_train_batch_size=2,  # Decrease batch size
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=1e-4
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

trainer.train()

# Step 6: Save the Fine-Tuned Model
model.save_pretrained("./fine-tuned-model-clean")

# Save the tokenizer to the same directory as the fine-tuned model
tokenizer.save_pretrained("./fine-tuned-model-clean")
