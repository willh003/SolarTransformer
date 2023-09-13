import csv
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import pandas as pd
from datetime import datetime
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from solar import get_radiation_prior

"""
idea: treat each day as a "word"
    - this means we project from 24 hours to high dimensional space, back to 24 hours
    - alternatively, treat each hour as a word
    - or, project back to 1 hour but use 24 hour words (questionable)
    - need positional encoding: use day of year/day of month/month of year
issues:
    - loss should be KL divergence, not mse (because mse says its terrible if it lags slightly)
    - lots of lag; how to fix?
    - potential idea: just predict total daily use, instead of each hour use for the next day
    """

class SeriesTransformer(pl.LightningModule):
    
    def __init__(self, input_length, channels=512, output_len = 1, lr=1e-3):
        super().__init__()
        self.channels = channels
        self.training_step_outputs = []
        self.validation_step_outputs = []


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=8,
            dim_feedforward=4 * channels,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels,
            nhead=8,
            dim_feedforward=4 * channels,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=8)
        self.input_embedding = nn.Linear(in_features = input_length, out_features=channels)
        self.output_embedding = nn.Linear(in_features = input_length, out_features=channels)
        self.prediction_layer = nn.Linear(in_features = channels, out_features=output_len) # maybe make this 1 
        self.pos_encoding = self.generate_pos_encoding(channels)
        self.lr = lr
        self.step = 0

    def training_step(self, x, batch_idx):
        src, trg = x
        pred = self.forward(x)
        #loss = nn.functional.mse_loss(pred.flatten(), trg.permute(1,0,2).sum(dim=2).flatten())
        loss = nn.functional.mse_loss(pred, trg.permute(1,0,2))
        logs={"train_loss": loss}
 
        batch_dictionary={
            #REQUIRED: It ie required for us to return "loss"
            "loss": loss,
             
            #optional for batch logging purposes
            "log": logs,
 
            # info to be used at epoch end 
            "batch": batch_idx,
        }
        #self.training_step_outputs.append(batch_dictionary)
        self.step += 1
        if self.step % 100 == 0:
            self.log("train_loss", loss,prog_bar=True)
        return loss
    
    def validation_step(self, x, batch_idx):
        src, trg = x
        pred = self.forward(x)
        #loss = nn.functional.mse_loss(pred.flatten(), trg.permute(1,0,2).sum(dim=2).flatten())
        loss = nn.functional.mse_loss(pred, trg.permute(1,0,2))
        out = {
            "src": src,
            "trg": trg,
            "pred": pred,
            "loss": loss
        }

        self.validation_step_outputs.append(out)
        
        return out

    def on_validation_epoch_end(self):
        #  the function is called after every epoch is completed

        # calculating average loss  
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
 
        # creating log dictionary
        tensorboard_logs = {'loss': avg_loss}
 
        epoch_dictionary={
            # required
            'loss': avg_loss,
             
            # for logging purposes
            'log': tensorboard_logs}
        
        self.log("val_loss", avg_loss,prog_bar=True)
        self.validation_step_outputs.clear()
        return epoch_dictionary

    def on_train_epoch_end(self):
        return
        #  the function is called after every epoch is completed
 
        # calculating average loss  
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
 
        # creating log dictionary
        tensorboard_logs = {'loss': avg_loss}
 
        epoch_dictionary={
            # required
            'loss': avg_loss,
             
            # for logging purposes
            'log': tensorboard_logs}
        self.training_step_outputs.clear()
        self.log("train_loss", avg_loss,prog_bar=True)
        return epoch_dictionary

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        src, trg = x
        encoded_src = self.encode(src)
        out = self.decode(encoded_src, trg)
        return out

    def encode(self, x):
        """
        x: (B, input_length)
        """
        embedding = self._embed_and_position(x, self.input_embedding, self.pos_encoding)
        encoding = self.encoder(embedding.permute(1,0,2)) 
        return encoding
    
    def decode(self, mem, y):
        """
        mem: (input_length, B, embed_dim)
        y: (B, tgt_length)
        """
        B = y.size(0)
        tgt_length = y.size(1)
        tgt = self._embed_and_position(y, self.output_embedding, self.pos_encoding).permute(1,0,2)
        tgt_mask = self.get_tgt_mask(tgt_length)#.expand(B,tgt_length, tgt_length)

        embedding = self.decoder(tgt, mem, tgt_mask=tgt_mask) 
        out = self.prediction_layer(embedding)
        return out
        

    def _embed_and_position(self, x, embedding_function, pos_encoding):
        """
        Embed and add to positional encoding a vector x using an embedding function and positional encoding
        """
        
        embedding = embedding_function(x)

        new_embedding = embedding + pos_encoding.unsqueeze(0).expand(embedding.size())
        return new_embedding



    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1).cuda() # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask

    def generate_pos_encoding(self, channels):
        encoding = torch.zeros(channels).cuda()
        for i in range(channels):
            if i % 2 == 0:
                encoding[i] = np.sin(i / (10000 ** (2 * i / channels)))
            else:
                encoding[i] = np.cos(i / (10000 ** (2 * i / channels)))
        return encoding


class SeriesDataset(torch.utils.data.Dataset):
    def __init__(self, df, inp_length, tgt_length, tgt_horizon, mem_horizon, feature_col='energy'):
        data = torch.as_tensor(df[feature_col].values, dtype=torch.float32)
        self.data = data[:-(len(data) % inp_length)]
        
        self.mem_horizon = mem_horizon
        self.tgt_horizon = tgt_horizon
        self.input_length = inp_length
        self.tgt_ˇlength = tgt_length
    
    def initialize_data(self, data):
        all_sequences = []
        for start in range(len(data) - self.mem_horizon - self.tgt_horizon):
            all_sequences.append(data[start:start+self.mem_horizon + self.tgt_horizon])
        return torch.as_tensor(all_sequences).float()

    def __len__(self):
        return len(self.data) - self.mem_horizon*self.input_length - self.tgt_horizon*self.input_length
    
    def __getitem__(self, i):
        src_end = i + self.mem_horizon * self.input_length
        src = self.data[i: src_end].view( self.mem_horizon, self.input_length) 
        # shift tgt left by 1 (so its input is last output of src)
        tgt = self.data[src_end - 1: src_end - 1 + self.tgt_horizon * self.tgt_ˇlength].view(self.tgt_horizon, self.tgt_ˇlength) 
        return src, tgt
    
def load_csv(path):
    
    with open(path) as f:
        reader = csv.reader(f, delimiter=',', quotechar='|')
        next(reader)

        fst = next(reader)

        times = [datetime.strptime(fst[0], '%m/%d/%y %H:%M')]
        energies = [float(fst[1])]
        relative_times = [0]

        for row in reader:
            dt_object = datetime.strptime(row[0], '%m/%d/%y %H:%M')
            times.append(dt_object)
            time_difference = (dt_object - times[0]).total_seconds() / (24*60*60)
            relative_times.append(time_difference)
            energies.append(float(row[1]))

    data = pd.DataFrame({'time': times, 'energy': energies})
    return data

def add_date_cols(dataframe: pd.DataFrame, date_col: str = "timestamp"):
    """
    add time features like month, week of the year ...
    :param dataframe:
    :param date_col:
    :return:
    """

    dataframe[date_col] = pd.to_datetime(dataframe[date_col], format="%m/%d/%y %H:%M")

    dataframe["day_of_month"] = dataframe[date_col].dt.day / 31
    dataframe["day_of_year"] = dataframe[date_col].dt.dayofyear / 365
    dataframe["month"] = dataframe[date_col].dt.month / 12
    dataframe["week_of_year"] = dataframe[date_col].dt.isocalendar().week / 53
    dataframe["hour_of_day"] = dataframe[date_col].dt.hour / 24
    dataframe["year"] = (dataframe[date_col].dt.year - 2018) / 2
    dataframe["y"] = pd.to_numeric(dataframe["y"])

    return dataframe, ["day_of_month", "day_of_year", "month", "week_of_year", "hour_of_day" "year"]

def plot_batch_out(val_loader, model):
    test_point = next(iter(val_loader))
    src, trg = test_point
    model = model.cuda()
    with torch.no_grad():
        pred = model.forward((src.cuda(), trg.cuda()))

    fig, axes = plt.subplots(nrows=4, ncols=4)
    for i in range(16):
        hours = src[i].flatten()
        next_hours = trg[i].flatten()
        pred_range = range(len(hours), len(hours) + len(next_hours))

        ax = axes[i // 4][i % 4]
        ax.plot(range(len(hours)), hours )
        ax.plot(pred_range, next_hours)
        ax.plot(pred_range, pred[0][0].cpu())
    print(nn.functional.mse_loss(trg.cuda(), pred))


def plot_sgd_out(val_loader, model):
    fig, axes = plt.subplots(nrows=4, ncols=4)
    with torch.no_grad():
        for i in range(16):
            test_point = next(iter(val_loader))
            src, trg = test_point
            model = model.cuda()
            pred = model.forward((src.cuda(), trg.cuda()))
            hours = src.flatten()
            next_hours = trg.flatten()
            pred_range = range(len(hours), len(hours) + len(next_hours))

            ax = axes[i // 4][i % 4]
            ax.plot(range(len(hours)), hours )
            ax.plot(pred_range, next_hours)
            ax.plot(pred_range, pred[0][0].cpu())
    print(nn.functional.mse_loss(trg.cuda(), pred))
    plt.show()

def train():
    INP_LEN = 24
    OUTPUT_LEN = 24
    EMBED_DIM = 512
    MEM_HORIZON = 7
    FWD_HORIZON = 1
    EPOCHS = 1
    BATCH_SIZE = 1
    LR = 1e-4
    split = .8
    data = load_csv('energy.csv')
#    data, addl_features = add_date_cols(data)
    train_data = SeriesDataset(data[:int(len(data) * split)], INP_LEN, OUTPUT_LEN, FWD_HORIZON, MEM_HORIZON)
    val_data = SeriesDataset(data[int(len(data) * split):], INP_LEN, OUTPUT_LEN, FWD_HORIZON, MEM_HORIZON)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        num_workers=10,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        num_workers=10,
        shuffle=False,
    )

    model = SeriesTransformer(INP_LEN, EMBED_DIM, OUTPUT_LEN, LR)
    trainer = pl.Trainer(max_epochs=EPOCHS)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if BATCH_SIZE == 1:
        plot_sgd_out(val_loader, model)
    else:
        plot_batch_out(val_loader, model)



if __name__ == "__main__":
    train()