import torch
import pickle

model_path_1 = '/home/hhiromasa/code/Uni-Perceiver/asset/pt_in1k.pkl'
model_path_2 = '/home/hhiromasa/code/Uni-Perceiver/asset/pt_qqp.pkl'

#open both pickle files 
with open(model_path_1, 'rb') as f:
    model_1 = pickle.load(f)
    
with open(model_path_2, 'rb') as f:
    model_2 = pickle.load(f)

model_1_dict = model_1.state_dict()
model_2_dict = model_2.state_dict()

merged_model_dict = model_1_dict.copy() 
    
#merge the two models
for k, v in merged_model_dict.items():
    interpolated_weight = (model_1_dict[k] + model_2_dict[k]) * 0.5
    v.copy_(interpolated_weight)
    
# Save the merged model
torch.save(merged_model_dict, '/home/hhiromasa/code/Uni-Perceiver/asset/merged_model.pth')