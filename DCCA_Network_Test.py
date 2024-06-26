import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
#%%
from sklearn.datasets import load_iris
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from DCCA_Network import DccaNet
import pandas as pd
import random
from random import randint


#%%
def train_dcca(view_1_data, view_2_data, hidden_dims, output_dim, epochs, batch_size, lr):
    input_dim_1 = view_1_data.shape[1]
    input_dim_2 = view_2_data.shape[1]
    
    # Pad the smaller view with zeros
    max_num_samples = max(view_1_data.size(0), view_2_data.size(0))
    if view_1_data.size(0) < max_num_samples:
        padding = max_num_samples - view_1_data.size(0)
        view_1_data = torch.cat([view_1_data, torch.zeros(padding, view_1_data.size(1))], dim=0)
    elif view_2_data.size(0) < max_num_samples:
        padding = max_num_samples - view_2_data.size(0)
        view_2_data = torch.cat([view_2_data, torch.zeros(padding, view_2_data.size(1))], dim=0)
    
    model = DccaNet(input_dim_1, input_dim_2, hidden_dims, output_dim)
    criterion = nn.CosineSimilarity(dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for i in range(0, max_num_samples, batch_size):
            batch_x1 = view_1_data[i:i+batch_size]
            batch_x2 = view_2_data[i:i+batch_size]
            
            optimizer.zero_grad()
            
            x1, x2 = model(batch_x1, batch_x2)
            loss = -criterion(x1, x2).mean()
            
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

def test_dcca(model, view_1_data, view_2_data):
    model.eval()
    with torch.no_grad():
        x1, x2 = model(view_1_data, view_2_data)
        corr = torch.mean(torch.sum(x1 * x2, dim=1))
    return corr.item()

def pad_with_zeros(array1, array2):
    
    array1 = np.array(array1)
    array2 = np.array(array2)

    # Get the shapes of the arrays
    rows1, cols1 = array1.shape
    rows2, cols2 = array2.shape
    
    max_rows = max(rows1, rows2)
    max_cols = max(cols1, cols2)
    
    padded_array1 = np.zeros((max_rows, max_cols))
    padded_array2 = np.zeros((max_rows, max_cols))

    padded_array1[:rows1, :cols1] = array1
    padded_array2[:rows2, :cols2] = array2

    return padded_array1, padded_array2

#%% load the data

main_data_path = 'Dataset/dataFilt.csv'
pathways_data_path = 'Dataset/kegg_legacy_ensembl.csv'

my_data = pd.read_csv(main_data_path)
my_pathways_data = pd.read_csv(pathways_data_path)

dataFrame = pd.DataFrame(my_pathways_data)

#%%
#pathway_1_key =  ['ENSG00000280830', 'ENSG00000101444', 'ENSG00000168710', 'ENSG00000158467', 'ENSG00000123505', 'ENSG00000149089', 'ENSG00000145692', 'ENSG00000160200', 'ENSG00000129596', 'ENSG00000116761', 'ENSG00000130816', 'ENSG00000119772', 'ENSG00000088305', 'ENSG00000142182', 'ENSG00000145293', 'ENSG00000120053', 'ENSG00000125166', 'ENSG00000104951', 'ENSG00000134333', 'ENSG00000166800', 'ENSG00000171989', 'ENSG00000111716', 'ENSG00000166796', 'ENSG00000151224', 'ENSG00000168906', 'ENSG00000038274', 'ENSG00000128309', 'ENSG00000099810', 'ENSG00000116984', 'ENSG00000135094', 'ENSG00000102172', 'ENSG00000116649', 'ENSG00000198650', 'ENSG00000107614']
#pathway_2_key = ['ENSG00000105829', 'ENSG00000177951', 'ENSG00000113734', 'ENSG00000108587', 'ENSG00000108433', 'ENSG00000265808', 'ENSG00000092531', 'ENSG00000132639', 'ENSG00000099940', 'ENSG00000143740', 'ENSG00000104915', 'ENSG00000135604', 'ENSG00000117758', 'ENSG00000124222', 'ENSG00000136874', 'ENSG00000168818', 'ENSG00000178750', 'ENSG00000106089', 'ENSG00000099365', 'ENSG00000111450', 'ENSG00000166900', 'ENSG00000103496', 'ENSG00000162236', 'ENSG00000135823', 'ENSG00000079950', 'ENSG00000170310', 'ENSG00000171045', 'ENSG00000053501', 'ENSG00000139190', 'ENSG00000220205', 'ENSG00000049245', 'ENSG00000117533', 'ENSG00000168899', 'ENSG00000124333', 'ENSG00000118640', 'ENSG00000151532', 'ENSG00000100568', 'ENSG00000106636'] 

pathway_1_key = ['ENSG00000196839', 'ENSG00000164742', 'ENSG00000143199', 'ENSG00000078295', 'ENSG00000138031', 'ENSG00000129467', 'ENSG00000173175', 'ENSG00000174233', 'ENSG00000121281', 'ENSG00000155897', 'ENSG00000162104', 'ENSG00000156110', 'ENSG00000170222', 'ENSG00000239900', 'ENSG00000185100', 'ENSG00000035687', 'ENSG00000106992', 'ENSG00000004455', 'ENSG00000162433', 'ENSG00000154027', 'ENSG00000140057', 'ENSG00000151360', 'ENSG00000116748', 'ENSG00000116337', 'ENSG00000133805', 'ENSG00000198931', 'ENSG00000138363', 'ENSG00000171302', 'ENSG00000156136', 'ENSG00000114956', 'ENSG00000197594', 'ENSG00000154269', 'ENSG00000138185', 'ENSG00000054179', 'ENSG00000168032', 'ENSG00000197217', 'ENSG00000187097', 'ENSG00000197586', 'ENSG00000188833', 'ENSG00000189283', 'ENSG00000262473', 'ENSG00000119125', 'ENSG00000137198', 'ENSG00000100938', 'ENSG00000163655', 'ENSG00000164116', 'ENSG00000152402', 'ENSG00000061918', 'ENSG00000070019', 'ENSG00000132518', 'ENSG00000101890', 'ENSG00000143774', 'ENSG00000165704', 'ENSG00000106348', 'ENSG00000178035', 'ENSG00000125877', 'ENSG00000239672', 'ENSG00000011052', 'ENSG00000243678', 'ENSG00000103024', 'ENSG00000103202', 'ENSG00000112981', 'ENSG00000172113', 'ENSG00000143156', 'ENSG00000169418', 'ENSG00000159899', 'ENSG00000125458', 'ENSG00000116981', 'ENSG00000185013', 'ENSG00000076685', 'ENSG00000122643', 'ENSG00000135318', 'ENSG00000205309', 'ENSG00000164978', 'ENSG00000165609', 'ENSG00000170502', 'ENSG00000128050', 'ENSG00000138801', 'ENSG00000198682', 'ENSG00000112541', 'ENSG00000128655', 'ENSG00000115252', 'ENSG00000123360', 'ENSG00000154678', 'ENSG00000186642', 'ENSG00000172572', 'ENSG00000152270', 'ENSG00000065989', 'ENSG00000184588', 'ENSG00000105650', 'ENSG00000113448', 'ENSG00000138735', 'ENSG00000132915', 'ENSG00000133256', 'ENSG00000095464', 'ENSG00000156973', 'ENSG00000185527', 'ENSG00000139053', 'ENSG00000205268', 'ENSG00000171408', 'ENSG00000073417', 'ENSG00000113231', 'ENSG00000160191', 'ENSG00000178921', 'ENSG00000143627', 'ENSG00000067225', 'ENSG00000198805', 'ENSG00000138035', 'ENSG00000101868', 'ENSG00000014138', 'ENSG00000062822', 'ENSG00000106628', 'ENSG00000077514', 'ENSG00000175482', 'ENSG00000177084', 'ENSG00000100479', 'ENSG00000148229', 'ENSG00000115350', 'ENSG00000068654', 'ENSG00000125630', 'ENSG00000171453', 'ENSG00000186184', 'ENSG00000137054', 'ENSG00000066379', 'ENSG00000181222', 'ENSG00000047315', 'ENSG00000102978', 'ENSG00000144231', 'ENSG00000099817', 'ENSG00000100142', 'ENSG00000168002', 'ENSG00000163882', 'ENSG00000105258', 'ENSG00000005075', 'ENSG00000228049', 'ENSG00000285437', 'ENSG00000147669', 'ENSG00000177700', 'ENSG00000148606', 'ENSG00000013503', 'ENSG00000186141', 'ENSG00000168495', 'ENSG00000132664', 'ENSG00000113356', 'ENSG00000121851', 'ENSG00000100413', 'ENSG00000161980', 'ENSG00000128059', 'ENSG00000198056', 'ENSG00000146143', 'ENSG00000147224', 'ENSG00000229937', 'ENSG00000101911', 'ENSG00000143363', 'ENSG00000167325', 'ENSG00000171848', 'ENSG00000048392', 'ENSG00000183463', 'ENSG00000158125']
pathway_2_key = ['ENSG00000075673', 'ENSG00000105675', 'ENSG00000186009', 'ENSG00000152234', 'ENSG00000110955', 'ENSG00000165629', 'ENSG00000099624', 'ENSG00000124172', 'ENSG00000159199', 'ENSG00000227590', 'ENSG00000135390', 'ENSG00000154518', 'ENSG00000169020', 'ENSG00000241468', 'ENSG00000167283', 'ENSG00000116459', 'ENSG00000167863', 'ENSG00000154723', 'ENSG00000241837', 'ENSG00000071553', 'ENSG00000033627', 'ENSG00000185344', 'ENSG00000105929', 'ENSG00000117410', 'ENSG00000185883', 'ENSG00000159720', 'ENSG00000147614', 'ENSG00000113732', 'ENSG00000171130', 'ENSG00000114573', 'ENSG00000116039', 'ENSG00000147416', 'ENSG00000155097', 'ENSG00000143882', 'ENSG00000100554', 'ENSG00000131100', 'ENSG00000250565', 'ENSG00000128524', 'ENSG00000136888', 'ENSG00000213760', 'ENSG00000151418', 'ENSG00000047249', 'ENSG00000006695', 'ENSG00000166260', 'ENSG00000014919', 'ENSG00000138495', 'ENSG00000131143', 'ENSG00000131055', 'ENSG00000178741', 'ENSG00000135940', 'ENSG00000111775', 'ENSG00000156885', 'ENSG00000126267', 'ENSG00000160471', 'ENSG00000164919', 'ENSG00000266251', 'ENSG00000161281', 'ENSG00000112695', 'ENSG00000115944', 'ENSG00000131174', 'ENSG00000170516', 'ENSG00000127184', 'ENSG00000176340', 'ENSG00000187581', 'ENSG00000179091', 'ENSG00000107902', 'ENSG00000198899', 'ENSG00000228253', 'ENSG00000198804', 'ENSG00000198712', 'ENSG00000198938', 'ENSG00000198727', 'ENSG00000198888', 'ENSG00000198763', 'ENSG00000198840', 'ENSG00000198886', 'ENSG00000212907', 'ENSG00000198786', 'ENSG00000198695', 'ENSG00000125356', 'ENSG00000130414', 'ENSG00000174886', 'ENSG00000131495', 'ENSG00000170906', 'ENSG00000189043', 'ENSG00000185633', 'ENSG00000128609', 'ENSG00000184983', 'ENSG00000267855', 'ENSG00000119421', 'ENSG00000139180', 'ENSG00000004779', 'ENSG00000183648', 'ENSG00000140990', 'ENSG00000090266', 'ENSG00000119013', 'ENSG00000065518', 'ENSG00000136521', 'ENSG00000165264', 'ENSG00000099795', 'ENSG00000166136', 'ENSG00000147684', 'ENSG00000109390', 'ENSG00000151366', 'ENSG00000023228', 'ENSG00000158864', 'ENSG00000213619', 'ENSG00000164258', 'ENSG00000168653', 'ENSG00000145494', 'ENSG00000115286', 'ENSG00000110717', 'ENSG00000167792', 'ENSG00000178127', 'ENSG00000160194', 'ENSG00000180817', 'ENSG00000138777', 'ENSG00000073578', 'ENSG00000117118', 'ENSG00000143252', 'ENSG00000204370', 'ENSG00000110719', 'ENSG00000184076', 'ENSG00000237528', 'ENSG00000127540', 'ENSG00000156467', 'ENSG00000010256', 'ENSG00000140740', 'ENSG00000169021', 'ENSG00000173660', 'ENSG00000233954', 'ENSG00000164405']


#%%
view_1 = my_data[my_data['index'].isin(pathway_1_key)]
view_2 = my_data[my_data['index'].isin(pathway_2_key)]

view_1_data = view_1.drop(['index'], axis=1)
view_1_data = view_1_data.values
print(view_1_data.shape)
#%%

view_2_data = view_2.drop(['index'], axis=1)
view_2_data = view_2_data.values
print(view_2_data.shape)

rows1, cols1 = view_1_data.shape
rows2, cols2 = view_2_data.shape
    
#print(rows1)
#print(cols1)
max_samples = max(len(view_1_data), len(view_2_data))

#%%
# Pad the smaller view with zeros
padded_view_1, padded_view_2 = pad_with_zeros(view_1_data, view_2_data)


#%% Split the data into train and test sets
train_view_1, test_view_1 = padded_view_1[:82], padded_view_1[55:]
train_view_2, test_view_2 = padded_view_2[:82], padded_view_2[55:]

train_view_1 = torch.from_numpy(train_view_1).float()
train_view_2 = torch.from_numpy(train_view_2).float()
test_view_1 = torch.from_numpy(test_view_1).float()
test_view_2 = torch.from_numpy(test_view_2).float()
# Dummy data

#print(train_view_1.shape)
#print(train_view_2.shape)

#%%
hidden_dims = [32, 16]
output_dim = 8
epochs = 100
batch_size = 800
lr = 0.0005

model = train_dcca(train_view_1, train_view_2, hidden_dims, output_dim, epochs, batch_size, lr)

# Test the model
test_corr = test_dcca(model, test_view_1, test_view_2)
print(f"Test correlation: {test_corr:.4f}")

#%%