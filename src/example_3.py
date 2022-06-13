import Data
import SupervisedModel
from Correction import corriger


model = SupervisedModel.SupervisedModel()

# model.load_model("model_0", "LabelEncoder_0")
# print(model.model.summary())

data = Data.Data()
for i in range(1, 3+1):
    data.load_json(f"../data/recording_{i}.json")

model.set_train_test_data(data.get_DataFrame(), shuffle_test=False)
model.clean_data()
model.preprocess_data()

model.initialize_model_from_data()

model.train()

prediction = model.predict_on_test_set(True)

predicted_text = "".join([e if e != "space" else " " for e in prediction])

print(predicted_text)

print(corriger(predicted_text))

