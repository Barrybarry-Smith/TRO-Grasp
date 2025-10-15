# V5: from v2, remove the v_mask, every node is valid
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class NaiveMLP(torch.nn.Module):
    def __init__(
        self, 
        z_dim, 
        out_dim, 
        hidden_dims
    ):
        super().__init__()
        c_wide_out_cnt = z_dim
        self.layers = nn.ModuleList()
        c_in = z_dim
        for i, c_out in enumerate(hidden_dims):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(c_in, c_out),
                    nn.LayerNorm(c_out),
                    nn.LeakyReLU(),
                )
            )
            c_wide_out_cnt += c_out
            c_in = c_out
        self.out_fc0 = nn.Linear(c_wide_out_cnt, out_dim)
        return

    def forward(self, x):
        f_list = [x]
        for l in self.layers:
            x = l(x)
            f_list.append(x)
        f = torch.cat(f_list, -1)
        y = self.out_fc0(f)
        return y


class GraphLayer(nn.Module):
    def __init__(
        self,
        v_object_dim,
        v_robot_dim,
        t_embed_dim,
        e_or_dim,
        e_rr_dim,
        c_atten_head
    ) -> None:
        super().__init__()

        self.c_atten_head = c_atten_head

        # PE binding
        self.v_object_binding_fc = nn.Sequential(
            nn.Linear(v_object_dim + t_embed_dim, v_object_dim),
            nn.SiLU()
        )
        self.v_robot_binding_fc = nn.Sequential(
            nn.Linear(v_robot_dim + t_embed_dim, v_robot_dim),
            nn.SiLU()
        )
        self.e_or_binding_fc = nn.Sequential(
            nn.Linear(e_or_dim + t_embed_dim, e_or_dim),
            nn.SiLU()
        )
        self.e_rr_binding_fc = nn.Sequential(
            nn.Linear(e_rr_dim + t_embed_dim, e_rr_dim),
            nn.SiLU()
        )

        # OR Attention
        self.or_query_fc = nn.Linear(v_robot_dim, e_or_dim)
        self.or_key_fc = nn.Linear(v_object_dim, e_or_dim)
        self.or_value_fc = nn.Sequential(
            nn.Linear(v_object_dim + v_robot_dim + e_or_dim, e_or_dim),
            nn.SiLU(),
            nn.Linear(e_or_dim, e_or_dim)
        )
        self.or_r_self = nn.Sequential(
            nn.Linear(v_robot_dim, e_or_dim),
            nn.SiLU()
        )
        self.or_out = nn.Sequential(
            nn.Linear(2 * e_or_dim, v_robot_dim),
            nn.SiLU()
        )
        self.or_edge_out = nn.Sequential(
            nn.Linear(2 * e_or_dim, e_or_dim),
            nn.SiLU(),
            nn.Linear(e_or_dim, e_or_dim)
        )

        # RR Attention
        self.rr_query_fc = nn.Linear(v_robot_dim, e_rr_dim)
        self.rr_key_fc = nn.Linear(v_robot_dim, e_rr_dim)
        self.rr_value_fc = nn.Sequential(
            nn.Linear(2 * v_robot_dim + e_rr_dim, e_rr_dim),
            nn.SiLU(),
            nn.Linear(e_rr_dim, e_rr_dim)
        )
        self.rr_r_self = nn.Sequential(
            nn.Linear(v_robot_dim, e_rr_dim),
            nn.SiLU()
        )
        self.rr_out = nn.Sequential(
            nn.Linear(2 * e_rr_dim, v_robot_dim),
            nn.SiLU()
        )
        self.rr_edge_out = nn.Sequential(
            nn.Linear(2 * e_rr_dim, e_rr_dim),
            nn.SiLU(),
            nn.Linear(e_rr_dim, e_rr_dim)
        )

    def forward(
        self,
        object_node_f,
        robot_node_f,
        or_edge_f,
        rr_edge_f,
        t_embed  # [B, 200]
    ):
        B, L, P, F = or_edge_f.shape
        
        ## Binding
        object_node_f = self.v_object_binding_fc(
            torch.cat([object_node_f, t_embed[:, None, :].expand(-1, P, -1)], dim=-1)
        )  # [B, P, N_o]
        robot_node_f = self.v_robot_binding_fc(
            torch.cat([robot_node_f, t_embed[:, None, :].expand(-1, L, -1)], dim=-1)
        )  # [B, L, N_r]
        or_edge_f = self.e_or_binding_fc(
            torch.cat([or_edge_f, t_embed[:, None, None, :].expand(-1, L, P, -1)], dim=-1)
        )  # [B, L, P, E_or]
        rr_edge_f = self.e_rr_binding_fc(
            torch.cat([rr_edge_f, t_embed[:, None, None, :].expand(-1, L, L, -1)], dim=-1)
        )  # [B, L, L, E_rr]

        ## OR attention
        or_query = self.or_query_fc(robot_node_f)[:, :, None, :].expand(-1, -1, P, -1)  # [B, L, P, E_or]
        or_key = self.or_key_fc(object_node_f)[:, None, :, :].expand(-1, L, -1, -1)  # [B, L, P, E_or]

        or_attn = or_query * or_key  # [B, L, P, E_or]
        or_attn = or_attn.reshape(B, L, P, -1, self.c_atten_head)
        or_attn = or_attn.sum(-1, keepdim=True) / np.sqrt(self.c_atten_head * 3.0)
        
        or_attn = torch.softmax(or_attn, 2)
        or_attn = or_attn.expand(-1, -1, -1, -1, self.c_atten_head).reshape(B, L, P, -1)  # [B, L, P, E_or]

        or_edge_o = object_node_f[:, None, :, :].expand(-1, L, -1, -1)  # [B, L, P, N_o]
        or_edge_r = robot_node_f[:, :, None, :].expand(-1, -1, P, -1)   # [B, L, P, N_r]
        or_value = self.or_value_fc(
            torch.cat([
                or_edge_o, or_edge_r, or_edge_f
            ], dim=-1)
        )  # [B, L, P, E_or]

        agg_or = (or_attn * or_value).sum(2)  # [B, L, E_or]
        self_or = self.or_r_self(robot_node_f)  # [B, L, E_or]
        robot_node_f = agg_or + self_or

        # global
        robot_node_f_pooling = robot_node_f.max(1, keepdim=True).values.expand(-1, L, -1)  # [B, L, E_or]
        robot_node_f = self.or_out(torch.cat([robot_node_f, robot_node_f_pooling], dim=-1))  # [B, L, N_r]

        # update or edge
        or_edge_f_pool_1 = or_edge_f.mean(1, keepdim=True).expand(-1, L, -1, -1)
        or_edge_f_pool_2 = or_edge_f.mean(2, keepdim=True).expand(-1, -1, P, -1)
        or_edge_f_pool = (or_edge_f_pool_1 + or_edge_f_pool_2) / 2.0   # [B, L, P, E_or]
        or_value = self.or_edge_out(
            torch.cat([or_value, or_edge_f_pool], dim=-1)
        )

        ## RR attention
        rr_query = self.rr_query_fc(robot_node_f)[:, :, None, :].expand(-1, -1, L, -1)  # [B, L, L, E_rr]
        rr_key = self.rr_key_fc(robot_node_f)[:, None, :, :].expand(-1, L, -1, -1)  # [B, L, L, E_rr]

        rr_attn = rr_query * rr_key  # [B, L, L, E_rr]
        rr_attn = rr_attn.reshape(B, L, L, -1, self.c_atten_head)
        rr_attn = rr_attn.sum(-1, keepdim=True) / np.sqrt(self.c_atten_head * 3.0)
        
        rr_attn = torch.softmax(rr_attn, 2)
        rr_attn = rr_attn.expand(-1, -1, -1, -1, self.c_atten_head).reshape(B, L, L, -1)  # [B, L, L, E_rr]

        rr_edge_self = robot_node_f[:, None, :, :].expand(-1, L, -1, -1)  # [B, L, L, N_r]
        rr_edge_neighbor = robot_node_f[:, :, None, :].expand(-1, -1, L, -1)   # [B, L, L, N_r]
        rr_value = self.rr_value_fc(
            torch.cat([
                rr_edge_self, rr_edge_neighbor, rr_edge_f
            ], dim=-1)
        )  # [B, L, L, E_rr]

        agg_rr = (rr_attn * rr_value).sum(2)  # [B, L, E_rr]
        self_rr = self.rr_r_self(robot_node_f)  # [B, L, E_rr]
        robot_node_f = agg_rr + self_rr

        # global
        robot_node_f_pooling = robot_node_f.max(1, keepdim=True).values.expand(-1, L, -1)  # [B, L, E_rr]
        robot_node_f = self.rr_out(torch.cat([robot_node_f, robot_node_f_pooling], dim=-1))  # [B, L, N_r]

        # update rr edge
        rr_edge_f_pool_1 = rr_edge_f.mean(1, keepdim=True).expand(-1, L, -1, -1)
        rr_edge_f_pool_2 = rr_edge_f.mean(2, keepdim=True).expand(-1, -1, L, -1)
        rr_edge_f_pool = (rr_edge_f_pool_1 + rr_edge_f_pool_2) / 2.0   # [B, L, P, E_rr]
        rr_value = self.or_edge_out(
            torch.cat([rr_value, rr_edge_f_pool], dim=-1)
        )

        return robot_node_f, or_value, rr_value


class GraphDenoiser(torch.nn.Module):

    def __init__(
        self,
        # Initial Config
        M: int = 1000,
        object_patch: int = 25,
        max_link_node: int = 25,
        # Embedding
        t_embed_dim: int = 200,
        # Input Encoder
        V_object_dims: list = [3, 1, 64],
        V_robot_dims: list = [3, 3, 64],
        E_or_dims: list = [3, 3],
        E_rr_dims: list = [3, 3],
        # Backbone
        v_conv_dim=384,
        e_conv_dim=384,
        c_atten_head=32,
        num_layers=6,
        v_out_hidden_dims=[256, 128],
        e_out_hidden_dims=[256, 128],
        se3_out_dim=[3, 3]
    ) -> None:
        super().__init__()

        self.M = M
        self.object_patch = object_patch
        self.max_link_node = max_link_node

        # Position Embedding
        self.t_embed_dim = t_embed_dim
        self.register_buffer("t_embed", self.sinusoidal_embedding(self.M, self.t_embed_dim))

        # Node Embedding
        self.V_object_dims = V_object_dims
        self.V_robot_dims = V_robot_dims
        self.V_object_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, v_conv_dim) for in_dim in self.V_object_dims]
        )
        self.V_robot_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, v_conv_dim) for in_dim in self.V_robot_dims]
        )

        # Edge Embedding
        self.E_or_dims = E_or_dims
        self.E_rr_dims = E_rr_dims
        self.E_or_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, e_conv_dim) for in_dim in self.E_or_dims]
        )
        self.E_rr_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, e_conv_dim) for in_dim in self.E_rr_dims]
        )

        # Backbone
        self.v_conv_dim = v_conv_dim
        self.e_conv_dim = e_conv_dim
        self.c_atten_head = c_atten_head
        self.num_layers = num_layers
        
        self.graph_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.graph_layers.append(
                GraphLayer(
                    v_object_dim=v_conv_dim,
                    v_robot_dim=v_conv_dim,
                    t_embed_dim=t_embed_dim,
                    e_or_dim=e_conv_dim,
                    e_rr_dim=e_conv_dim,
                    c_atten_head=c_atten_head
                )
            )
        self.v_robot_wide_fc = nn.Linear(v_conv_dim * (1 + num_layers), v_conv_dim)

        # Output layers
        self.se3_out_dim = se3_out_dim
        self.v_robot_output_module = nn.ModuleList()

        for f_dim in self.se3_out_dim:
            self.v_robot_output_module.append(
                NaiveMLP(
                    f_dim + self.v_conv_dim, 
                    f_dim,
                    v_out_hidden_dims
                )
            )

    @staticmethod
    def sinusoidal_embedding(n, d):
        # Returns the standard positional embedding
        embedding = torch.zeros(n, d)
        wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
        wk = wk.reshape((1, d))
        t = torch.arange(n).reshape((n, 1))
        embedding[:, ::2] = torch.sin(t * wk[:, ::2])
        embedding[:, 1::2] = torch.cos(t * wk[:, ::2])
        return embedding

    def _encoder_(self, x, in_dims, in_layers):
        # Encode node and edge
        assert len(in_dims) == len(in_layers)
        cur = 0
        feat = None
        for i, d in enumerate(in_dims):
            partial = x[..., cur : cur + d]
            out = in_layers[i](partial)
            if feat is None:
                feat = out
            else:
                feat = feat + out
            cur += d
        return feat

    def forward(
        self, 
        V_O, 
        noisy_V_R,
        noisy_E_OR,
        noisy_E_RR,
        t,
    ):

        # Position Embedding
        B = t.shape[0]
        t_embed = self.t_embed[t]  # [B, 200]

        # Initial Encoding
        object_node_f = self._encoder_(
            V_O,
            self.V_object_dims,
            self.V_object_in_layers
        )  # [B, P, F]
        robot_node_f = self._encoder_(
            noisy_V_R,
            self.V_robot_dims,
            self.V_robot_in_layers
        )  # [B, L, F]
        or_edge_f = self._encoder_(
            noisy_E_OR,
            self.E_or_dims,
            self.E_or_in_layers
        )  # [B, L, P, F]
        rr_edge_f = self._encoder_(
            noisy_E_RR,
            self.E_rr_dims,
            self.E_rr_in_layers
        )  # [B, L, L, F]

        noisy_robot_node_f_list = [robot_node_f]
        noisy_or_edge_f_list = [or_edge_f]
        noisy_rr_edge_f_list = [rr_edge_f]

        for layer in self.graph_layers:
            robot_node_f, or_edge_f, rr_edge_f = layer(
                object_node_f,
                robot_node_f,
                or_edge_f,
                rr_edge_f,
                t_embed
            )
            noisy_robot_node_f_list.append(robot_node_f)
            noisy_or_edge_f_list.append(or_edge_f)
            noisy_rr_edge_f_list.append(rr_edge_f)

        update_robot_node_f = self.v_robot_wide_fc(
            torch.cat(noisy_robot_node_f_list, dim=-1)
        )  # [B, L, F]

        # Output
        v_robot_pred = []
        cur = 0
        for layer_id, f_dim in enumerate(self.se3_out_dim):
            v_robot_pred.append(
                self.v_robot_output_module[layer_id](
                    torch.cat([update_robot_node_f, noisy_V_R[..., cur : cur + f_dim]], dim=-1)
                )
            )
            cur += f_dim
        v_robot_pred = torch.cat(v_robot_pred, dim=-1)
        return v_robot_pred
        