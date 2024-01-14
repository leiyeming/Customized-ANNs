import sys
sys.path.append(r'\GitHub\Customized-ANNs\codes')
from C_ANN_fucs import *

# set random seeds for reproducibility
torch.manual_seed(1)

# True: train a new model; False: load a trained model
model_train = True

# check if a GPU is available use GPU otherwise use CPU
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

loc = '\GitHub\Customized-ANNs\data\s2_data\'
scenario=2
scenario=str(scenario)
# Load train, test, and two prediction datasets from CSV files.
# need to combine all datasets to get the same dummy variables (categorical variables one-hot encoding and continue variables normalization)

feature_names = ['lon', 'lat', 'yr', 'vess', 'tech_1', 'tech_2']
cat_cols = ['vess', 'yr']
# tr = pd.read_csv('train.csv')
types_dict = {'yr': object}
tr = pd.read_csv(loc + 'train.csv', dtype=types_dict)
lon_lat = tr[["lon", "lat"]]   # normalize lon and lat
X_tr, y_tr = tr[feature_names], tr['catch'].values

ts = pd.read_csv(loc+'test.csv', dtype=types_dict)
X_ts, y_ts = ts[feature_names], ts['catch'].values

pred_data = pd.read_csv(loc+'predict_dataset.csv', dtype=types_dict)
X_pred, y_pred= pred_data[feature_names], pred_data['catch'].values

pred_data = pd.read_csv(loc+'predict_all_location.csv', dtype=types_dict)
X_pred_all, y_pred_all= pred_data[feature_names], pred_data['catch'].values

# combine all datasets
train_row=X_tr.shape[0]
test_row=X_ts.shape[0]
pred_row=X_pred.shape[0]
pred_all_row=X_pred_all.shape[0]

new_data=pd.concat([pd.DataFrame(X_tr),pd.DataFrame(X_ts),pd.DataFrame(X_pred),pd.DataFrame(X_pred_all)], axis=0)
new_data = new_data.reset_index(drop=True)

# normalize continue variables (lon and lat in this case)
new_data[["lon", "lat"]] = lon_lat_norm(lon_lat, new_data[["lon", "lat"]])
# Conver tcategorical features to dummy features.

### onehotencoder categorical variables only
# Identify which columns are categorical
MLP_1_col = ['lon', 'lat']
MLP_1_col.extend('yr_' + tr['yr'].unique())

cat_cols = ['vess','yr']
# Create a OneHotEncoder object for categorical variables
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
new_data_encoded = encoder.fit_transform(new_data[cat_cols])
# Assign the column names to the encoded data
column_names = encoder.get_feature_names_out(cat_cols)
new_data_encoded = pd.DataFrame(new_data_encoded, columns=column_names)

# Combine categorical data and the continuous data
new_data = pd.concat([pd.DataFrame(new_data_encoded), new_data.drop(cat_cols, axis=1)], axis=1)

#Convert data from numpy arrays to PyTorch tensors.
y_tr, y_ts, y_pred, y_pred_all  = n2t(y_tr).unsqueeze(1), n2t(y_ts).unsqueeze(1),  n2t(y_pred).unsqueeze(1), n2t(y_pred_all).unsqueeze(1)

# split into 2 parts (one with for MLP_1 and the other for MLP_2)
new_data_nl,new_data_l= new_data[MLP_1_col], new_data.drop(MLP_1_col, axis=1)

# split into train, test, and prediction datasets
X_tr_nl, X_ts_nl, X_pred_nl, X_pred_all_nl= split_data(new_data_nl, train_row,test_row,pred_row,pred_all_row)
X_tr_l, X_ts_l, X_pred_l, X_pred_all_l= split_data(new_data_l, train_row,test_row,pred_row,pred_all_row)

## define the model,
# Since we only use MLP for the spatial-temporal variables, we only need to define one MLP and combine the output with the other variables.
neuron=[64,64,64]
print("neuron:", neuron)
NN_str = "ANN_ST_"
for j in neuron:
    NN_str = NN_str + str(j) + "_"
NN_str = NN_str[:-1]
# layer=len(neuron)

net=MLP_ANN_ST(X_tr_nl.shape[1],neuron,X_tr_l.shape[1])
net=net.to(device)


## define the optimizer and loss function
# optimizer = SGD(net.parameters(), lr=1e-2)
optimizer = Adam(net.parameters())
criterion = nn.MSELoss()

## define the dataset and early stopping
dataset = DatasetWrapper_ANN_S_ST(X_tr_nl,X_tr_l, torch.log(y_tr))
# early stopping
early_stopping = EarlyStopping(patience=300, verbose=False,loc=loc,NN_str=NN_str)
######### model training  #########
if model_train:
    train_ANN_S_ST(net, optimizer, dataset, batch_size=100, nepochs=3000, val_nl=X_tr_nl, val_l=X_tr_l,
                val_y=y_tr, device=device, scenario=scenario, NN_str=NN_str, early_stopping=early_stopping,
                criterion=criterion)
# load best model
model_path = loc + "model/" + str(NN_str) + '.pt'
net.load_state_dict(torch.load(model_path))

######### prediction  #########
with torch.no_grad():
    net.to(torch.device('cpu'))
    pred_tr = torch.exp(net(X_tr_nl, X_tr_l))
    pred_ts = torch.exp(net(X_ts_nl, X_ts_l))
    pred_pred = torch.exp(net(X_pred_nl, X_pred_l))
    pred_pred_all = torch.exp(net(X_pred_all_nl, X_pred_all_l))
R2 = r2_score(y_tr, pred_tr)
mse_train = mean_squared_error(y_tr, pred_tr)
mse_test = mean_squared_error(y_ts, pred_ts)
mse_visited = mean_squared_error(y_pred, pred_pred)
mse_all = mean_squared_error(y_pred_all, pred_pred_all)

print("mse_train:", mse_train, "mse_test:",
      mse_test, "R2:", R2,
      "mse_visited:", mse_visited,
      "mse_all:", mse_all)
