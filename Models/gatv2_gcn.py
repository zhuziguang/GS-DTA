import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GATv2_GCN_CNN_BiLSTM_Transformer(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=128, embed_dim=128, output_dim=128, dropout=0.2,
                 nhead=4, transformer_layers=2):

        super(GATv2_GCN_CNN_BiLSTM_Transformer, self).__init__()

        self.n_output = n_output
        self.gatv2_conv1 = GATv2Conv(num_features_xd, num_features_xd, heads=10)
        self.gcn_conv1 = GCNConv(num_features_xd * 10, num_features_xd * 10)
        
        self.gcn_conv2 = GCNConv(num_features_xd, num_features_xd)
        self.gcn_conv3 = GCNConv(num_features_xd, num_features_xd * 2)
        self.gcn_conv4 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        
        self.fc_g1 = torch.nn.Linear(num_features_xd * 10 * 2 + num_features_xd * 4 * 2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # CNN for protein sequence
        self.conv_xt1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=3, padding=1)
        self.pool_xt1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.lstm_xt = nn.LSTM(input_size=embed_dim, hidden_size=output_dim // 2, num_layers=2, dropout=dropout, bidirectional=True)

        # Transformer
        transformer_layer = TransformerEncoderLayer(d_model=output_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(transformer_layer, num_layers=transformer_layers)

        # Combined layers
        self.fc1 = nn.Linear(output_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        # GATv2Conv branch
        x1 = self.gatv2_conv1(x, edge_index)
        x1 = self.relu(x1)
        x1 = self.gcn_conv1(x1, edge_index)
        x1 = self.relu(x1)
        x1 = torch.cat([gmp(x1, batch), gap(x1, batch)], dim=1)

        # GCNConv branch with three layers
        x2 = self.gcn_conv2(x, edge_index)
        x2 = self.relu(x2)
        x2 = self.gcn_conv3(x2, edge_index)
        x2 = self.relu(x2)
        x2 = self.gcn_conv4(x2, edge_index)
        x2 = self.relu(x2)
        x2 = torch.cat([gmp(x2, batch), gap(x2, batch)], dim=1)

        # Concatenate both branches
        x = torch.cat((x1, x2), dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        # Protein input feed-forward
        target = data.target
        embedded_xt = self.embedding_xt(target)  # [batch_size, seq_len, embed_dim]

        # Reshape for CNN
        conv_xt = self.conv_xt1(embedded_xt)  # Apply CNN
        conv_xt = self.pool_xt1(F.relu(conv_xt))  # Apply pooling and activation

        # Reshape for LSTM
        conv_xt = conv_xt.permute(2, 0, 1)  # Flatten the output for LSTM
        lstm_out, (h_n, c_n) = self.lstm_xt(conv_xt)

        # Prepare LSTM output for Transformer
        transformer_out = self.transformer_encoder(lstm_out)

        # Since we're interested in the overall representation, take the mean across the seq_len dimension
        transformer_out = transformer_out.mean(dim=0)

        # Concatenate graph and protein representations
        xc = torch.cat((x, transformer_out), 1)

        # Additional fully connected layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
