
import torch.nn as nn

class LBAD(nn.Module):
    def __init__(self, neurons, activation, drop_out_p):
        super(LBAD, self).__init__()
        self.neurons = neurons
        self.activ = activation()
        self.drop_out_p = drop_out_p

        self.w1 = nn.Linear(self.neurons, self.neurons)
        self.bn1 = nn.BatchNorm1d(self.neurons)
        self.dropout = nn.Dropout(p=self.drop_out_p)

    def forward(self, x):
        x = self.w1(x)
        x = self.bn1(x)
        x = self.activ(x)
        x = self.dropout(x)

        return x

class Encoder2D(nn.Module):

    def __init__(self, latent_dim, n_joints=17, activation=nn.ReLU):
        super(Encoder2D, self).__init__()
        self.latent_dim = latent_dim
        self.activation = activation
        self.n_joints = n_joints
        self.neurons = 1024
        self.name = "Encoder2D"
        self.drop_out_p = 0.5
        self.blocks = 2

        self.__build_model()

    def __build_model(self):

        self.enc_inp_block = nn.Sequential(
            nn.Linear(2*self.n_joints, self.neurons),  # expand features
            nn.BatchNorm1d(self.neurons),
            self.activation(),
            nn.Dropout(p=self.drop_out_p)
        )

        self.LBAD_1 = LBAD(self.neurons, self.activation, self.drop_out_p)
        self.LBAD_2 = LBAD(self.neurons, self.activation, self.drop_out_p)
        if self.blocks > 1:
            self.LBAD_3 = LBAD(self.neurons, self.activation, self.drop_out_p)
            self.LBAD_4 = LBAD(self.neurons, self.activation, self.drop_out_p)

        self.fc = nn.Linear(self.neurons, self.latent_dim)

      

    def forward(self, x):
        x = x.reshape(-1, 2*self.n_joints)#x.view(-1, 2*self.n_joints)
        x = self.enc_inp_block(x)

        # to explore
        '''BaseLine'''
        residual = x
        x = self.LBAD_1(x)
        x = self.LBAD_2(x) + residual
        
        if self.blocks > 1:
            residual = x
            x = self.LBAD_3(x)
            x = self.LBAD_4(x) + residual
        

        return self.fc (x)


class Decoder3D(nn.Module):
    def __init__(self, latent_dim, n_joints=17, activation=nn.ReLU):
        super(Decoder3D, self).__init__()
        self.latent_dim = latent_dim
        self.activation = activation
        self.n_joints = n_joints
        self.neurons = 1024
        self.name = "Decoder3D"
        self.drop_out_p = 0.5
        self.blocks = 2
        self.__build_model()

    def __build_model(self):

        self.dec_inp_block = nn.Sequential(
            nn.Linear(self.latent_dim, self.neurons),
            nn.BatchNorm1d(self.neurons),
            self.activation(),
            nn.Dropout(p=self.drop_out_p)
        )

        self.LBAD_1 = LBAD(self.neurons, self.activation, self.drop_out_p)
        self.LBAD_2 = LBAD(self.neurons, self.activation, self.drop_out_p)
        if self.blocks > 1:
            self.LBAD_3 = LBAD(self.neurons, self.activation, self.drop_out_p)
            self.LBAD_4 = LBAD(self.neurons, self.activation, self.drop_out_p)


        self.dec_out_block = nn.Sequential( nn.Linear(self.neurons, 3*self.n_joints),)

    def forward(self, x):
        x = x.view(-1, self.latent_dim)
        x = self.dec_inp_block(x)

        '''BaseLine'''
        residual = x
        x = self.LBAD_1(x)
        x = self.LBAD_2(x) + residual
        if self.blocks > 1:
            residual = x
            x = self.LBAD_3(x)
            x = self.LBAD_4(x) + residual
      
        x = self.dec_out_block(x)

        return x
    
    
    
    