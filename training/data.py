from datasets import load_dataset


dataset = load_dataset("wmt14", "de-en")

print(dataset['train'][0]['translation'])
print(dataset['train'][1])
print(dataset['train'][2])