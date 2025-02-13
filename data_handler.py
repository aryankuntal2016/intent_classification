
import pandas as pd

INTENT_DF = pd.read_csv("intent_data.csv")


INTENT_FREQ = INTENT_DF["intent"].value_counts()
INTENT_TO_IDX_DICT = {intent: idx for idx, intent in enumerate(INTENT_DF["intent"].unique())}
IDX_TO_INTENT_DICT = {idx : intent for intent, idx in INTENT_TO_IDX_DICT.items()}

TEXT = list(INTENT_DF["text"])
LABELS = [INTENT_TO_IDX_DICT[x] for x in INTENT_DF["intent"]]
NUM_INTENTS = len(INTENT_TO_IDX_DICT)

