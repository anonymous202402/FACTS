import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_function(outputs, batch_y, model):
    loss = nn.MSELoss()(outputs, batch_y) + model.bop_img.get_orthogonality_loss()
    return loss

class BilinearOrthogonalProjector(nn.Module):

    def __init__(self, in_height: int, in_width: int, r_h: int, r_w: int):

        super().__init__()
        self.r_h = r_h
        self.r_w = r_w
        
  
        self.U = nn.Parameter(torch.empty(in_height, r_h))
        self.V = nn.Parameter(torch.empty(in_width, r_w))
        
   
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        projected = torch.einsum('...hw,hi,wj->...ij', x, self.U, self.V)
        

        flattened = projected.flatten(start_dim=-2)
        
        return flattened

    def get_orthogonality_loss(self) -> torch.Tensor:
 
        device = self.U.device
        
        # Create Identity matrices
        I_h = torch.eye(self.r_h, device=device)
        I_w = torch.eye(self.r_w, device=device)
        
        # Compute U^T U and V^T V
        u_prod = torch.matmul(self.U.T, self.U)
        v_prod = torch.matmul(self.V.T, self.V)
        
        # Compute Frobenius norm squared of the difference from Identity
        loss_u = torch.norm(u_prod - I_h, p='fro') ** 2
        loss_v = torch.norm(v_prod - I_w, p='fro') ** 2
        
        return loss_u + loss_v