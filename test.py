import pickle
from train_attributes import TOKENIZER
from data_handler import IDX_TO_INTENT_DICT
import torch


MODEL = None

with open("model.pkl", "rb") as f:
	MODEL = pickle.load(f)


def get_intent(text):
	test_encoding = TOKENIZER(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
	with torch.no_grad():
		output = MODEL(**test_encoding)
		predicted_id = torch.argmax(output.logits, dim=-1).item()

	predicted_label = IDX_TO_INTENT_DICT[predicted_id]

	return predicted_label


if __name__ == "__main__":

	text = "my VISA is rejected what can i do next"
	label = get_intent(text)

	print(label)







