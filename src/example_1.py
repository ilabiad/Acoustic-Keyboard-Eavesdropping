import Data
import SupervisedModel
from Correction import corriger
import UnsupervisedModel


data = Data.Data()
for i in range(1, 8+1):
    data.load_json(f"../data/recording_{i}.json")

model = UnsupervisedModel.UnsupervisedModel()

model.set_data(data.get_DataFrame())

model.cluster_data(30)
print(model.cluster_score())
predicted_text = model.train_and_predict()

print(predicted_text)
print(corriger(predicted_text))


