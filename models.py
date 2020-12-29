import torch.nn as nn
import torch.nn.functional as f
import torch
import numpy as np
import csv
import pandas as pd

class Model(nn.Module):
    def __init__(self, hidden_size, attn_dim, in_dim, out_dim, dropout=0.1):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.attn_dim = attn_dim
        self.lstm1 = nn.LSTM(in_dim, hidden_size, num_layers=1, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.q_proj = nn.Linear(hidden_size, attn_dim)
        self.k_proj = nn.Linear(hidden_size, attn_dim)
        self.v_proj = nn.Linear(hidden_size, attn_dim)
        self.attn_out = nn.Linear(attn_dim, hidden_size)
        self.prediction_out = nn.Linear(hidden_size, out_dim)
        self.attn_dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, lens, attn_mask):
        # x - (batch_size, player, time, dim)
        # attn_mask - (batch_size, player, time). 1 if should be masked, 0 else.
        # lens - 1-d tensor of sequence lengths

        #prepare data
        b_size, num_players, time, in_dim = x.shape
        x = x.reshape(-1, time, in_dim)
        lens_stretched = lens.unsqueeze(1).repeat(1, num_players).reshape(-1)

        # encode sequence with 1 layer lstm
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lens_stretched, batch_first=True)
        packed_encoded_input, _ = self.lstm1(packed_x)
        encoded_input, _ = nn.utils.rnn.pad_packed_sequence(packed_encoded_input, batch_first=True)
        encoded_input = encoded_input.reshape(b_size, num_players, time, self.hidden_size).permute(0, 2, 1, 3)

        # attention computation
        q, k, v = self.q_proj(encoded_input), self.k_proj(encoded_input), self.v_proj(encoded_input)
        attn_logit = torch.einsum('btqd,btkd->btqk', q, k) / np.sqrt(self.hidden_size)
        attn_dist = self.attn_dropout(f.softmax(attn_logit - 1e24*attn_mask.permute(0, 2, 1).unsqueeze(2).repeat(1, 1, num_players, 1), dim=-1))
        combined_vals = torch.einsum('btqk,btkd->btqd', attn_dist, v)
        attn_output = self.attn_out(combined_vals).permute(0, 2, 1, 3)
        
        #2 layer lstm over attention outputs
        packed_x2 = nn.utils.rnn.pack_padded_sequence(attn_output.reshape(-1, time, self.hidden_size), lens_stretched, batch_first=True)
        packed_lstm2_output, _ = self.lstm2(packed_x2)
        lstm2_output, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm2_output, batch_first=True)
        lstm2_output = lstm2_output.reshape(b_size, num_players, time, self.hidden_size)

        # final output
        predictions = self.prediction_out(lstm2_output)
        return predictions

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_files):
        super(Dataset).__init__()
        self.data_files = data_files
        self.data_frames = []
        for data_file in self.data_files:
            with open(data_file, 'r') as f:
                self.data_frames.append(pd.read_csv(f))
            print('loaded', data_file)
        self.plays = [sorted(set(data_frame['playId'])) for data_frame in self.data_frames]
        self.frames = [{play: sorted(set(self.data_frames[i].query('playId=={play}'.format(play=play))['frameId'])) for play in data_frame} for i, data_frame in enumerate(self.plays)]
        self.team_idxs = {'home': 0, 'away': 1, 'football': 2, 'unk': 3}
        self.start_idxs = [0]
        for item in self.plays:
            self.start_idxs.append(self.start_idxs[-1] + len(item))

    def __getitem__(self, key):
        # outputs shape (time, player, dim)
        # dim = 3, first two corridinates are x and y, the last is team label
        idx = 0
        while key >= self.start_idxs[idx]:
            idx += 1
        idx -= 1
        key -= self.start_idxs[idx]
        play = self.plays[idx][key]
        frames = self.frames[idx][play]
        xs = []
        ys = []
        teams = []
        for frame in frames:
            raw_data = self.data_frames[idx].query('playId=={play} & frameId=={frame}'.format(play=play, frame=frame))[['x', 'y', 'team']].values.tolist()
            x, y, team = list(zip(*raw_data))
            xs.append([item if not np.isnan(item) else 0.0 for item in x])
            ys.append([item if not np.isnan(item) else 0.0 for item in y])
            teams.append([self.team_idxs[item] if item is not None else 3 for item in team])
        max_len = max(map(len, xs))
        xs = [list(x) + [0.0]*(max_len - len(x)) for x in xs]
        ys = [list(y) + [0.0]*(max_len - len(y)) for y in ys]
        teams = [list(team) + [3]*(max_len - len(team)) for team in teams]
        xs = np.stack(xs, axis=0)
        ys = np.stack(ys, axis=0)
        teams = np.stack(teams, axis=0)
        return np.stack([xs, ys, teams], axis=-1)
    
    def __len__(self):
        return sum(map(len, self.plays))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device', device)
    #training procedure
    train_dataset = Dataset(['data/standardized_week_%d_by_play.csv' % (i) for i in range(1, 16)])
    val_dataset = Dataset(['data/standardized_week_%d_by_play.csv' % (i) for i in range(16, 18)])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = Model(256, 256, 34, 4).float()
    #add embeddings for team, ball label to the model
    model.team_embeddings = nn.Embedding(4, 32)
    model = model.to(device)
    model.train()
    epochs = 10

    optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    step = 0
    for epoch in range(epochs):
        for item in train_loader:
            # first two items in input data are x, y the third item is a scalar representing team, get the embedding for this scalar and concat with x and y
            item = item.to(device)
            input_data = torch.cat([item[:, :, :, :2], model.team_embeddings(item[:, :, :, 2].long())], dim=-1).float().permute(0, 2, 1, 3).contiguous()
            attn_mask = (item[:, :, :, 2] == 3).float().permute(0, 2, 1)
            # outputs 4 things for each time step and each player:
            # the first 2 are the predicted mean x any y and the last 2 are their standard deviations
            output = model(input_data[:, :, :-1, :], torch.tensor([input_data.shape[2]-1]).to(device), attn_mask[:, :, :-1])
            truth = input_data[:, :, 1:, :2]
            prediction_means, prediction_stds = output[:, :, :, :2], output[:, :, :, 2:]
            # MLE loss function that takes into account variance prediction, can be negative (assumes independent gaussians, this is a strong assumption, but outputing a whole covariance matrix would be hard)
            loss = torch.mean(torch.sum(torch.sum(((prediction_means - truth) / torch.exp(prediction_stds))**2 + 2 * prediction_stds, dim=-1) * (1 - attn_mask[:, :, :-1]), dim=1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            print(epoch, step, loss.item())
            step += 1


    
