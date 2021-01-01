import torch.nn as nn
import torch.nn.functional as f
import torch
import numpy as np
import csv
import pandas as pd
import os
import wandb

class Model(nn.Module):
    def __init__(self, hidden_size, attn_dim, in_dim, out_dim, head2_dim, dropout=0.1):
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
        self.second_head = nn.Linear(hidden_size, head2_dim)
    
    def forward(self, x, lens, attn_mask):
        # x - (batch_size, player, time, dim)
        # attn_mask - (batch_size, player, time). 1 if should be masked, 0 else.
        # lens - 1-d tensor of sequence lengths

        #prepare data
        b_size, num_players, time, in_dim = x.shape
        x = x.reshape(-1, time, in_dim)
        lens_stretched = lens.unsqueeze(1).repeat(1, num_players).reshape(-1)

        # encode sequence with 1 layer lstm
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lens_stretched, batch_first=True, enforce_sorted=False)
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
        packed_x2 = nn.utils.rnn.pack_padded_sequence(attn_output.reshape(-1, time, self.hidden_size), lens_stretched, batch_first=True, enforce_sorted=False)
        packed_lstm2_output, _ = self.lstm2(packed_x2)
        lstm2_output, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm2_output, batch_first=True)
        lstm2_output = lstm2_output.reshape(b_size, num_players, time, self.hidden_size)

        # final output
        predictions = self.prediction_out(lstm2_output)
        return predictions

team_idxs = {'home': 0, 'away': 1, 'football': 2, 'unk': 3}
direction_idxs = {'left': 0, 'right': 1, 'unk': 2}
position_idxs = {'CB': 0, 'DB': 1, 'DE': 2, 'DL': 3, 'FB': 4, 
                'FS': 5, 'HB': 6, 'ILB': 7, 'LB': 8, 'MLB': 9, 
                'NT': 10, 'OLB': 11, 'QB': 12, 'RB': 13, 'S': 14, 
                'SS': 15, 'TE': 16, 'WR': 17, 'unk': 18}

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
        self.start_idxs = [0]
        for item in self.plays:
            self.start_idxs.append(self.start_idxs[-1] + len(item))

    def __getitem__(self, key):
        # outputs shape (batch=1, time, player, dim)
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
        speeds = []
        accelerations = []
        orientations = []
        directions = []
        positions = []
        for frame in frames:
            raw_data = self.data_frames[idx].query('playId=={play} & frameId=={frame}'.format(play=play, frame=frame))[['x', 'y', 'team', 's', 'o', 'a', 'playDirection', 'position']].values.tolist()
            x, y, team, s, o, a, dir, pos = list(zip(*raw_data))
            xs.append([item if not np.isnan(item) else 0.0 for item in x])
            ys.append([item if not np.isnan(item) else 0.0 for item in y])
            speeds.append([item if not np.isnan(item) else 0.0 for item in s])
            accelerations.append([item if not np.isnan(item) else 0.0 for item in a])
            orientations.append([item if not np.isnan(item) else 0.0 for item in o])
            teams.append([team_idxs[item] if item is not None else team_idxs['unk'] for item in team])
            directions.append([direction_idxs[item] if item is not None else direction_idxs['unk'] for item in dir])
            positions.append([position_idxs[item] if item is not None and type(item) is str else position_idxs['unk'] for item in pos])
        max_len = max(map(len, xs))
        xs = [list(x) + [0.0]*(max_len - len(x)) for x in xs]
        ys = [list(y) + [0.0]*(max_len - len(y)) for y in ys]
        speeds = [list(s) + [0.0]*(max_len - len(s)) for s in speeds]
        accelerations = [list(a) + [0.0]*(max_len - len(a)) for a in accelerations]
        orientations = [list(o) + [0.0]*(max_len - len(o)) for o in orientations]
        teams = [list(team) + [team_idxs['unk']]*(max_len - len(team)) for team in teams]
        directions = [list(direction) + [direction_idxs['unk']]*(max_len - len(direction)) for direction in directions]
        positions = [list(position) + [position_idxs['unk']]*(max_len - len(position)) for position in positions]
        xs = np.stack(xs, axis=0)
        ys = np.stack(ys, axis=0)
        speeds = np.stack(speeds, axis=0)
        accelerations = np.stack(accelerations, axis=0)
        orientations = np.stack(orientations, axis=0)
        teams = np.stack(teams, axis=0)
        return np.stack([xs, ys, speeds, accelerations, orientations, teams, directions, positions], axis=-1)
    
    def __len__(self):
        return sum(map(len, self.plays))

def pad_batch(data_batch):
    # data_batch - list of data tensors from dataset
    max_time = max(map(lambda x: x.shape[1], data_batch))
    max_players = max(map(lambda x: x.shape[2], data_batch))
    batch_padded = []
    for batch_item in data_batch:
        pad1 = torch.zeros((batch_item.shape[0], max_time-batch_item.shape[1], batch_item.shape[2], batch_item.shape[3]))
        pad1[:, :, :, 5] = team_idxs['unk']
        pad1[:, :, :, 6] = direction_idxs['unk']
        pad1[:, :, :, 7] = position_idxs['unk']
        pad2 = torch.zeros(batch_item.shape[0], max_time, max_players-batch_item.shape[2], batch_item.shape[3])
        pad2[:, :, :, 5] = team_idxs['unk']
        pad2[:, :, :, 6] = direction_idxs['unk']
        pad2[:, :, :, 7] = position_idxs['unk']
        batch_padded.append(torch.cat([torch.cat([batch_item, pad1], dim=1), pad2], dim=2))
    return torch.cat(batch_padded, dim=0)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device', device)
    # os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(project="nfl-big-data-bowl-2021")
    #training procedure
    train_dataset = Dataset(['data/standardized_week_%d_by_play.csv' % (i) for i in range(1, 2)])
    val_dataset = Dataset(['data/standardized_week_%d_by_play.csv' % (i) for i in range(2, 3)])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = Model(256, 256, 69, 10, 2).float()
    #add embeddings for team, ball label to the model
    model.team_embeddings = nn.Embedding(len(team_idxs), 32)
    model.dir_embeddings = nn.Embedding(len(direction_idxs), 32)
    model.pos_embeddings = nn.Embedding(len(position_idxs), 32)
    # model.load_state_dict(torch.load('trajectory_model.pkl', map_location='cpu'), strict=False)
    model = model.to(device)
    wandb.watch(model)
    model.train()
    config = wandb.config
    config.epochs = 10
    config.bsize = 8
    config.val_steps = 16

    optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    step = 0
    best_val_loss = float('inf')
    curr_batch = []
    for epoch in range(config.epochs):
        for item in train_loader:
            # collect batch
            curr_batch.append(item)
            if len(curr_batch) < config.bsize:
                continue

            #pad batch
            lens = torch.tensor(list(map(lambda x: x.shape[1], curr_batch))).int().to('cpu')
            items = pad_batch(curr_batch)
            # first two items in input data are x, y the third item is a scalar representing team, get the embedding for this scalar and concat with x and y
            items = items.to(device)
            input_data = torch.cat([items[:, :, :, :5], model.team_embeddings(items[:, :, :, 5].long()) + model.dir_embeddings(items[:, :, :, 6].long()), model.pos_embeddings(items[:, :, :, 7].long())], dim=-1).float().permute(0, 2, 1, 3).contiguous()
            attn_mask = (items[:, :, :, 5] == team_idxs['unk']).float().permute(0, 2, 1)
            # outputs 4 things for each time step and each player:
            # the first 2 are the predicted mean x any y and the last 2 are their standard deviations
            output = model(input_data[:, :, :-1, :], lens-1, attn_mask[:, :, :-1])
            truth = input_data[:, :, 1:, :5]
            prediction_means, prediction_stds = output[:, :, :, :5], output[:, :, :, 5:]
            # MLE loss function that takes into account variance prediction, can be negative (assumes independent gaussians, this is a strong assumption, but outputing a whole covariance matrix would be hard)
            loss = torch.mean(torch.sum(torch.sum(((prediction_means - truth) / torch.exp(prediction_stds))**2 + 2 * prediction_stds, dim=-1) * (1 - attn_mask[:, :, :-1]) * (1 - attn_mask[:, :, 1:]), dim=1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            curr_batch = []

            if step % 100 == 0:
                # eval model
                model.eval()
                val_batch = []
                total_val_loss = 0.0
                val_step = 0
                for val_item in val_loader:
                    val_batch.append(val_item)
                    if len(val_batch) < config.bsize:
                        continue
                    val_lens = torch.tensor(list(map(lambda x: x.shape[1], val_batch))).int().to('cpu')
                    val_items = pad_batch(val_batch)

                    val_items = val_items.to(device)
                    val_input_data = torch.cat([val_items[:, :, :, :5], model.team_embeddings(val_items[:, :, :, 5].long()) + model.dir_embeddings(val_items[:, :, :, 6].long()), model.pos_embeddings(val_items[:, :, :, 7].long())], dim=-1).float().permute(0, 2, 1, 3).contiguous()
                    val_attn_mask = (val_items[:, :, :, 5] == team_idxs['unk']).float().permute(0, 2, 1)
                    val_output = model(val_input_data[:, :, :-1, :], val_lens-1, val_attn_mask[:, :, :-1])
                    val_truth = val_input_data[:, :, 1:, :5]
                    val_prediction_means, val_prediction_stds = val_output[:, :, :, :5], val_output[:, :, :, 5:]
                    val_loss = torch.mean(torch.sum(torch.sum(((val_prediction_means - val_truth) / torch.exp(val_prediction_stds))**2 + 2 * val_prediction_stds, dim=-1) * (1 - val_attn_mask[:, :, :-1]) * (1 - val_attn_mask[:, :, 1:]), dim=1))
                    total_val_loss += val_loss
                    val_batch = []
                    val_step += 1
                    if val_step >= config.val_steps:
                        break
                total_val_loss /= config.val_steps
                print('epoch: {epoch}, step: {step}, train loss: {train_loss}, val loss: {val_loss}'.format(epoch=epoch, step=step, train_loss=loss.item(), val_loss=total_val_loss.item()))
                wandb.log({'epoch': epoch, 'step': step, 'train_loss': loss.item(), 'val_loss': total_val_loss.item()})
                # save best model
                if total_val_loss < best_val_loss:
                    print('new best model. saving ...')
                    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'trajectory_model.pkl'))
                    best_val_loss = total_val_loss
                    print('saved')
                model.train()
            
            step += 1


    
