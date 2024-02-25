
import copy
from utils import positionalencoding2d
import torch
import torch.nn.functional as F
from torch import nn, Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Transformer(nn.Module):

    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, video_frames, batch_size):
        super().__init__()
        self.batch_size = batch_size
        
        self.alpha = 0.7
        self.beta = 1 - self.alpha

        self.decoder_output_from_previous = None
        self.tracker_decoder_output_from_previous = None

        self.video_frames = video_frames
        
        self.counter = 0

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        #tracker_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)########
        #self.tracker_decoder = TransformerDecoder(tracker_decoder_layer, num_decoder_layers)##########

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    
    def forward(self, images, object_querys):


        encoder_output = self.encoder(images)

        """
        if self.counter == self.video_frames:
            print("MOST NONE")
            self.decoder_output_from_previous = None
            self.tracker_decoder_output_from_previous = None
            self.counter = 0

            
        self.counter += 1
        #print(self.counter)

        if self.decoder_output_from_previous != None:


            object_querys = (self.alpha * object_querys + self.beta * self.decoder_output_from_previous.detach()).to(device)
            object_querys = torch.sigmoid(object_querys).to(device)

            #decoder_input_track = (self.alpha * decoder_input_track + self.beta * self.tracker_decoder_output_from_previous.detach()).to(device)
            #decoder_input_track = torch.sigmoid(decoder_input_track).to(device)
        """
        #print(object_querys.shape)

        decoder_output = self.decoder(encoder_output, object_querys).to(device)

        ########self.decoder_output_from_previous = decoder_output.squeeze(0).clone().to(device)

        #tracker_output = self.tracker_decoder(encoder_output, decoder_output.squeeze(0)).to(device)
        #self.tracker_decoder_output_from_previous = tracker_output.squeeze(0).clone().to(device)


        #print(self.counter)
        return decoder_output.squeeze(0)#, tracker_output.squeeze(0)
    
    """
    def forward(self, images, object_querys):


        encoder_output = self.encoder(images)

        
        if self.counter == self.video_frames:
            pass
            #print("MOST NONE")
            #self.decoder_output_from_previous = None
            #self.counter = 0
            #self.counter -= 2*self.batch_size
            
        self.counter += 1
        #print(self.counter)

        if self.decoder_output_from_previous != None:
            pass


            #object_querys = (self.alpha * object_querys + self.beta * self.decoder_output_from_previous.detach()).to(device)
            #object_querys = torch.sigmoid(object_querys).to(device)
        
        #print(object_querys.shape)
        
        decoder_output = self.decoder(encoder_output, object_querys).to(device)
        #print(decoder_output.shape)
        #self.decoder_output_from_previous = decoder_output.squeeze(0).clone().to(device)
        #print(self.counter)
        return decoder_output.squeeze(0)
    
    """
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, images):
        output = images

        for layer in self.layers:
            output = layer(output)

        return output

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, encoder_output, object_querys):
        output = object_querys

        for layer in self.layers:
            output = layer(output, encoder_output)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.attn_weight = None
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def with_pos_embed(self, tensor):
        return tensor + positionalencoding2d(*tensor.shape)

    def forward(self, images):

        q = k = self.with_pos_embed(images)
        #images2 = self.self_attn(q, k, value=images)[0]
        images2, self.attn_weight = self.self_attn(q, k, value=images)
        images = images + self.dropout1(images2)
        images = self.norm1(images)
        images2 = self.linear2(self.dropout(self.activation(self.linear1(images))))
        images = images + self.dropout2(images2)
        images = self.norm2(images)
        return images

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def with_pos_embed(self, tensor):
        return tensor + positionalencoding2d(*tensor.shape)

    def forward(self, output, encoder_output):
        q = k = self.with_pos_embed(output)
        output2 = self.self_attn(q, k, value=output)[0]
        output = output + self.dropout1(output2)
        output = self.norm1(output)
        output2 = self.multihead_attn(query=self.with_pos_embed(output),
                                   key=self.with_pos_embed(encoder_output),
                                   value=encoder_output)[0]
        output = output + self.dropout2(output2)
        output = self.norm2(output)
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = output + self.dropout3(output2)
        output = self.norm3(output)
        return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
