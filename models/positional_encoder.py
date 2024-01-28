'''Module to produce a positional encoding matrix.
'''

import math
import torch


class PositionalEncoder:
    '''Sine and Cosine positional encoder module.
    '''

    def __init__(self, token_length, output_dim, n=10000, device='cpu'):
        '''Param init.
        '''

        self.device = device
        self.token_length = int(token_length)
        self.output_dim = int(output_dim)
        self.n = n
        self.P = torch.zeros((self.token_length, self.output_dim), requires_grad=False).to(self.device)


    def calc_pos_embedding(self):
        '''Calculate the positional embedding matrix.
        '''
        k = torch.arange(start=0, end=self.token_length, requires_grad=False).to(self.device)
        i = torch.arange(start=0, end=math.ceil(self.output_dim/2),requires_grad=False).to(self.device)

        denominators = torch.pow(self.n, torch.div(torch.mul(i, 2), self.output_dim)).to(self.device)

        stacked_denominators = denominators.reshape(1,-1).repeat_interleave(self.token_length, dim=0)

        values = torch.div(k, stacked_denominators.transpose(0,1)).to(self.device)

        self.P.transpose(0,1)[::2] = torch.sin(values).to(self.device)

        try:
            self.P.transpose(0,1)[1::2] = torch.cos(values).to(self.device)

        except RuntimeError:
            self.P.transpose(0,1)[1::2] = torch.cos(values[:-1]).to(self.device)

        return None


    def __call__(self):

        self.calc_pos_embedding()

        return self.P.detach() #make sure there's no gradient flow to this positional encoding matrix.

