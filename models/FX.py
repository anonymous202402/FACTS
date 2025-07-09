import torch
import torch.nn as nn

from models.PathcTST_FX import Model, ContinuousLoss

class FractalGen(nn.Module):
    def __init__(self,
                 img_size_list,
                 embed_dim_list,
                 num_blocks_list,
                 num_heads_list,
                 generator_type_list,
                 dff_list,
                 e_layers_list,
                 fractal_level=0):
        super().__init__()

        # --------------------------------------------------------------------------
        # fractal specifics
        self.fractal_level = fractal_level
        self.num_fractal_levels = len(img_size_list)

        # --------------------------------------------------------------------------
        if self.fractal_level == 0:
            self.seq_len = img_size_list[0]
            self.pred_len = img_size_list[0]
            self.patch_len = 16
            self.stride = 8

        # Generator for the current level
        if generator_type_list[fractal_level] == "tst":
            generator = Model
        else:
            raise NotImplementedError

        self.generator = generator(
            seq_len=img_size_list[fractal_level],
            pred_len=img_size_list[fractal_level],
            patch_len=img_size_list[fractal_level]//img_size_list[fractal_level+1] * 2,
            stride=img_size_list[fractal_level]//img_size_list[fractal_level+1],
            d_model=embed_dim_list[fractal_level],
            dropout=0.1,
            factor=4,
            n_heads=num_heads_list[fractal_level],
            d_ff=dff_list[fractal_level],
            e_layers=e_layers_list[fractal_level],
        )

        # --------------------------------------------------------------------------
        # Build the next fractal level recursively
        if self.fractal_level < self.num_fractal_levels - 2:
            self.next_fractal = FractalGen(
                img_size_list=img_size_list,
                embed_dim_list=embed_dim_list,
                num_blocks_list=num_blocks_list,
                num_heads_list=num_heads_list,
                generator_type_list=generator_type_list,
                dff_list=dff_list,
                e_layers_list=e_layers_list,
                fractal_level=fractal_level+1
            )
        else:
            self.next_fractal = ContinuousLoss(
                cond_dim=embed_dim_list[fractal_level],
                depth=num_blocks_list[fractal_level + 1],
                width=embed_dim_list[fractal_level + 1],
                num_heads=num_heads_list[fractal_level + 1]
            )

    def forward(self, x):
        """
        Forward pass to get loss recursively.
        """
        x = self.generator(x)
        # Compute loss recursively from the next fractal level.
        x = self.next_fractal(x)

        return x

def fractalmar_in64(**kwargs):
    model = FractalGen(
        img_size_list=(96, 6, 1),
        embed_dim_list=(256, 128, 64),
        num_blocks_list=(6, 3, 1),
        num_heads_list=(4, 4, 2),
        dff_list=(1024, 512, 256),
        e_layers_list=(6, 3, 1),
        generator_type_list=("tst", "tst", "tst"),
        fractal_level=0,
        **kwargs)
    return model