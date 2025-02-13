from train_attributes import TRAINER, MODEL
import pickle

TRAINER.train()

with open("model.pkl", "wb") as f:
	pickle.dump(MODEL, f)





