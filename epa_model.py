import torch.nn as nn
import torch.nn.functional as f
import torch
import numpy as np
import csv
import pandas as pd
import os
import wandb
import transformers


class EPAModel(nn.Module):
    def __init__(self, hidden_dim, num_heads, dim_ff, num_layers, dropout=0.1):
        super(EPAModel, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_ff, dropout=dropout), num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x, attn_mask):
        # x - (batch_size, player, dim)
        # attn_mask - (batch_size, player). 1 if should be masked, 0 else.
        hidden = self.encoder(x.permute(1, 0, 2).contiguous(), src_key_padding_mask=attn_mask==1)
        return self.output(hidden)

team_idxs = {'home': 0, 'away': 1, 'football': 2, 'unk': 3}
direction_idxs = {'left': 0, 'right': 1, 'unk': 2}
position_idxs = {'CB': 0, 'DB': 1, 'DE': 2, 'DL': 3, 'FB': 4, 
                'FS': 5, 'HB': 6, 'ILB': 7, 'LB': 8, 'MLB': 9, 
                'NT': 10, 'OLB': 11, 'QB': 12, 'RB': 13, 'S': 14, 
                'SS': 15, 'TE': 16, 'WR': 17, 'unk': 18}
position_idxs = {'P': 0, 'FB': 1, 'DL': 2, 'FS': 3, 'CB': 4, 'TE': 5, 'DT': 6, 
                'DB': 7, 'RB': 8, 'LS': 9, 'MLB': 10, 'SS': 11, 'OLB': 12, 'QB': 13, 
                'DE': 14, 'ILB': 15, 'S': 16, 'LB': 17, 'NT': 18, 'HB': 19, 'K': 20, 'WR': 21, 
                'unk': 22}

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
        self.combined_frames = []
        for week in range(len(self.frames)):
            for play in self.frames[week]:
                for frame in self.frames[week][play]:
                    self.combined_frames.append((week, play, frame))

    def __getitem__(self, key):
        # outputs shape (batch=1, player, dim)
        # dim = 3, first two corridinates are x and y, the last is team label
        week, play, frame = self.combined_frames[key]
        raw_data = self.data_frames[week].query('playId=={play} & frameId=={frame}'.format(play=play, frame=frame))[['x', 'y', 'team', 's', 'o', 'a', 'playDirection', 'position', 'epa']].values.tolist()
        x, y, team, s, o, a, dir, pos, epa = list(zip(*raw_data))
        xs = np.array([item if not np.isnan(item) else 0.0 for item in x])
        ys = np.array([item if not np.isnan(item) else 0.0 for item in y])
        speeds = np.array([item if not np.isnan(item) else 0.0 for item in s])
        accelerations = np.array([item if not np.isnan(item) else 0.0 for item in a])
        orientations = np.array([item if not np.isnan(item) else 0.0 for item in o])
        teams = np.array([team_idxs[item] if item is not None else team_idxs['unk'] for item in team])
        directions = np.array([direction_idxs[item] if item is not None else direction_idxs['unk'] for item in dir])
        positions = np.array([position_idxs[item] if item is not None and type(item) is str else position_idxs['unk'] for item in pos])
        return np.stack([xs, ys, speeds, accelerations, orientations, teams, directions, positions], axis=-1), np.array(epa)
    
    def __len__(self):
        return len(self.combined_frames)

def pad_batch(data_batch_x, data_batch_y):
    # data_batch - list of data tensors from dataset
    max_players = max(map(lambda x: x.shape[1], data_batch_x))
    batch_padded_x = []
    batch_padded_y = []
    for batch_item in data_batch_x:
        pad1 = torch.zeros((batch_item.shape[0], max_players-batch_item.shape[1], batch_item.shape[2]))
        pad1[:, :, 6] = direction_idxs['unk']
        pad1[:, :, 7] = position_idxs['unk']
        batch_padded_x.append(torch.cat([batch_item, pad1], dim=1))
    for batch_item in data_batch_y:
        pad2 = torch.zeros((batch_item.shape[0], max_players-batch_item.shape[1]))
        batch_padded_y.append(torch.cat([batch_item, pad2], dim=1))
    return torch.cat(batch_padded_x, dim=0), torch.cat(batch_padded_y, dim=0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device', device)

    wandb.init(project="nfl-big-data-bowl-2021-epa-model")

    train_dataset = Dataset(['data/epa-included-week-%d.csv' % (i) for i in range(1, 2)])
    val_dataset = Dataset(['data/epa-included-week-%d.csv' % (i) for i in range(2, 3)])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = EPAModel(64, 4, 256, 6).float()
    #add embeddings for team, ball label to the model
    model.team_embeddings = nn.Embedding(len(team_idxs), 29)
    model.dir_embeddings = nn.Embedding(len(direction_idxs), 29)
    model.pos_embeddings = nn.Embedding(len(position_idxs), 30)

    model = model.to(device)
    wandb.watch(model)
    model.train()
    config = wandb.config
    config.epochs = 10
    config.bsize = 8
    config.val_steps = 16

    optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    lr_schedule = transformers.get_polynomial_decay_schedule_with_warmup(optim, num_warmup_steps=2000,
                                                                         num_training_steps=(config.epochs*(len(train_dataset)//(config.bsize))),
                                                                         power=1.0, lr_end=0.0)

    step = 0
    best_val_loss = float('inf')
    curr_batch_x = []
    curr_batch_y = []
    for epoch in range(config.epochs):
        for item_x, item_y in train_loader:
            # collect batch
            curr_batch_x.append(item_x)
            curr_batch_y.append(item_y)
            if len(curr_batch_x) < config.bsize:
                continue

            #pad batch
            items_x, items_y = pad_batch(curr_batch_x, curr_batch_y)
            # first two items in input data are x, y the third item is a scalar representing team, get the embedding for this scalar and concat with x and y
            items_x = items_x.to(device)
            truth = items_y.to(device)
            input_data = torch.cat([items_x[:, :, :5], model.team_embeddings(items_x[:, :, 5].long()) + model.dir_embeddings(items_x[:, :, 6].long()), model.pos_embeddings(items_x[:, :, 7].long())], dim=-1).float().contiguous()
            attn_mask = (items_x[:, :, 5] == team_idxs['unk']).float()
            # outputs 4 things for each time step and each player:
            # the first 2 are the predicted mean x any y and the last 2 are their standard deviations
            output = model(input_data, attn_mask)
            # MLE loss function that takes into account variance prediction, can be negative (assumes independent gaussians, this is a strong assumption, but outputing a whole covariance matrix would be hard)
            loss = torch.mean((output - truth)**2 * (1 - attn_mask))
            optim.zero_grad()
            loss.backward()
            optim.step()
            lr_schedule.step()
            curr_batch = []

            if step % 100 == 0:
                # eval model
                model.eval()
                val_batch_x = []
                val_batch_y = []
                total_val_loss = 0.0
                val_step = 0
                for val_item_x, val_item_y in val_loader:
                    val_batch_x.append(val_item_x)
                    val_batch_y.append(val_item_y)
                    if len(val_batch_x) < config.bsize:
                        continue
                    val_items_x, val_items_y = pad_batch(val_batch_x, val_batch_y)

                    val_items_x = val_items_x.to(device)
                    val_truth = val_items_y.to(device)
                    val_input_data = torch.cat([val_items_x[:, :, :5], model.team_embeddings(val_items_x[:, :, 5].long()) + model.dir_embeddings(val_items_x[:, :, 6].long()), model.pos_embeddings(val_items_x[:, :, 7].long())], dim=-1).float().contiguous()
                    val_attn_mask = (val_items_x[:, :, 5] == team_idxs['unk']).float()
                    val_output = model(val_input_data, val_attn_mask)
                    val_loss = torch.mean((val_output - val_truth)**2 * (1 - val_attn_mask))
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

