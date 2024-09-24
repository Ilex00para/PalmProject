import torch 

import torch.nn as nn


class positional_encoding(nn.Module):
    def __init__(self,d_model, max_len = 5000):
        super(positional_encoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:-1]
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1, activation="relu"):
        super(Encoder, self).__init__()
        self.dmodel = d_model
        self.encoder_blks = nn.Sequential()
        for i in range(num_encoder_layers):
            self.encoder_blks.add_module(f"EncoderBlock_{i}", nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation))
        self.pe = positional_encoding(d_model)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = src * torch.sqrt(self.dmodel)
        src = src + self.pe(src)
        output = self.encoder_blks(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        return output
    
class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout=0.1, activation="relu"):
        super(Decoder, self).__init__()
        
        self.encoder_blks = nn.Sequential()
        for i in range(num_decoder_layers):
            self.encoder_blks.add_module(f"DecoderBlock_{i}", nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation))
        self.pe = positional_encoding(d_model)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt * torch.sqrt(self.dmodel)
        tgt = tgt + self.pe(tgt)

        output = self.encoder_blks(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        
        return output
    
    

class TransformerEncoderDecoderGenerative(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.embedding_layer = nn.Linear(3, d_model)
        
        self.encoder = Encoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation)
        self.decoder = Decoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout, activation)

        self.ouptut_layer = nn.Linear(d_model, 3)

    def forward(self, src, tgt, len_generate, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        memory = self.encoder(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        for i in range(len_generate):
            tgt = self.embedding_layer(tgt)
            output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
            output = self.ouptut_layer(output)
            tgt = torch.cat((tgt, output), 1)

        return output