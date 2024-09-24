
from kan import *
from torch.utils.data import DataLoader
from Modules import BaseDatasetPalm


torch.set_default_dtype(torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


dataset = BaseDatasetPalm('male')

# %%
X1 = torch.from_numpy(dataset.X[:,:,15::9].reshape(dataset.X.shape[0], -1,1)) #get the average relative humidity data
X2 = torch.from_numpy(dataset.X[:,:,12::9].reshape(dataset.X.shape[0], -1,1)) #get the average relative humidity data
print(X1.shape, X2.shape)
X = torch.cat((X1, X2), dim=2) #concatenate the data
print(X.shape)
time = torch.from_numpy(np.linspace(0, 1, X1.shape[1])).reshape(1, 800, 1).broadcast_to(X.shape[0],800,1) #create a time tensor 
print(time.shape)
X = torch.cat((X, time), dim=2).reshape(-1,3) #concatenate the data and time tensor

y1 = torch.from_numpy(np.where(dataset.y == 0, 0, 1)).reshape(dataset.X.shape[0],1,1).broadcast_to(dataset.X.shape[0],800,1).reshape(-1,1) #get the labels
y2 = torch.where(y1 == 0, 1, 0) #get the inverse of the labels

y = torch.cat((y1, y2), dim=1) #concatenate the labels

print(X.shape, y.shape)



X_train = X[:int(0.8*len(X))]
X_test = X[int(0.8*len(X)):]
y_train = y[:int(0.8*len(y))]
y_test = y[int(0.8*len(y)):]

#training_dl = DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True)
#test_dl = DataLoader(list(zip(X_test, y_test)), batch_size=32, shuffle=False)

dataset = {'train_input':X_train.to(device=device, dtype=torch.float32), 
           'test_input':X_test.to(device=device, dtype=torch.float32), 
           'train_label':y_train.to(device=device, dtype=torch.float32), 
           'test_label':y_test.to(device=device, dtype=torch.float32)}

# As long as nothing is pruned everything works fine. But NaN occurs when the pruning thresholds are increased or the model is scaled up (some functions less important) in both cases pruning is triggered.


architecture = [3, 18, 2]
grid = 3

train_rmse = []
test_rmse = []



def inf(model, node_threshold=1e-2, edge_threshold=3e-2):
    print(f'\nNumber of active functions: {model.n_edge}\n')

    #model.attribute()
    
    edge_scores_layers =  len(model.edge_scores) if model.edge_scores is not None else None
    node_scores_layers = len(model.node_scores) if model.node_scores is not None else None

    #print(f'Edge Scores Layers: {model.edge_scores}')
    try:
        n_edges = [edge_scores.shape for edge_scores in model.edge_scores if edge_scores is not None]
    except TypeError:
        n_edges = None
    try:
        n_nodes = [node_scores.shape for node_scores in model.node_scores if node_scores is not None ]
    except TypeError:
        n_nodes = None
    
    act_edges = []
    act_nodes = []
    for edges, nodes in zip(model.edge_scores, model.node_scores):
        if edges is not None:
            act_edges.append(edges[edges > edge_threshold].shape[0])
        if nodes is not None:
            act_nodes.append(nodes[nodes > node_threshold].shape[0])
    
    masks =  [act_f.mask for act_f in model.act_fun]
    relative_masks = [(torch.sum(mask == 1)/mask.numel()).cpu().numpy() for mask in masks]


    print(f'\nEdge Scores Layers: {edge_scores_layers} with shapes: {n_edges} and {len(act_edges)} active edges\n')
    print(f'\nNode Scores Layers: {node_scores_layers} with shapes: {n_nodes} and {len(act_nodes)} active nodes\n')
    print(f'\nActive proportion of the Masks: {relative_masks}\n')
    print(f'\nGrid Size : {model.grid}')






def prune(self, node_th=1e-2, edge_th=3e-2, obs=True):
    #get activation functions
    if self.acts == None:
        self.get_act()

    if obs:
        print('Before edge pruning')
        inf(self)

    self.attribute()
    self.prune_edge(edge_th, log_history=False)

    if obs:
        print('After edge pruning')
        inf(self)
    
    self.forward(self.cache_data)
    self = self.prune_node(node_th, log_history=False)

    if obs:
        print('After node pruning')
        inf(self)
    self.log_history('prune')
    return self


# import torch

# with open('/home/u108-n256/PalmProject/NeuralNetwork_Testing/model/0.0_state', 'rb') as f:
#     model = pickle.load(f)

# model = torch.load('/home/u108-n256/PalmProject/NeuralNetwork_Testing/model/0.0_state')

#somehow the model.attribute() needs ti be called before pruning?????

model1 = MultKAN(width=architecture, grid=grid, k=3, seed=1, device=device)
test_loss = torch.inf
active_edges = torch.inf
lr=1e-5

df = {'Active Edges':[], 'Train Loss':[], 'Test Loss':[], 'Regularization':[]}

for i in range(1):
    results = model1.fit(dataset, steps=500, batch=32, lr=lr, loss_fn=torch.nn.MSELoss())
    

    df['Active Edges'].append(model1.n_edge)
    df['Train Loss'].append(np.mean(results['train_loss']))
    df['Test Loss'].append(np.mean(results['test_loss']))
    df['Regularization'].append(np.mean(results['reg']))

    print(f'Epoch: {i}\nActive edges: {model1.n_edge}\nLR:{lr}\n')        


df = pd.DataFrame(df)
df.to_csv('pruning_results.csv')




