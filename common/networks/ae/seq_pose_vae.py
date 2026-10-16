import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.transform import homoify, dehomoify
from networks.tgs.models.snowflake.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation, QueryAndGroup, GroupAll


class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction Layer using CUDA-optimized pointnet2_ops"""
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.group_all = group_all

        if group_all:
            self.grouper = GroupAll(use_xyz=True)
        else:
            self.grouper = QueryAndGroup(radius, nsample, use_xyz=True)

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:  xyz [B, 3, N], points [B, D, N]
        Output: new_xyz [B, 3, S], new_points [B, D', S]
        """
        # Ensure inputs are contiguous for CUDA operations
        xyz = xyz.contiguous()
        if points is not None:
            points = points.contiguous()

        xyz_t = xyz.permute(0, 2, 1).contiguous()  # [B, N, 3]

        if self.group_all:
            new_xyz_t = torch.zeros(xyz_t.size(0), 1, 3, device=xyz.device)
            # GroupAll expects: xyz (B, N, 3), new_xyz (ignored), features (B, C, N)
            # Returns: (B, 3 + C, 1, N)
            new_points = self.grouper(xyz_t, new_xyz_t, points)  # [B, 3+D, 1, N]
        else:
            # Farthest point sampling
            fps_idx = furthest_point_sample(xyz_t, self.npoint)  # [B, npoint]
            new_xyz_t = gather_operation(xyz, fps_idx).permute(0, 2, 1).contiguous()  # [B, npoint, 3]
            # QueryAndGroup expects: xyz (B, N, 3), new_xyz (B, npoint, 3), features (B, C, N)
            # Returns: (B, 3 + C, npoint, nsample)
            new_points = self.grouper(xyz_t, new_xyz_t, points)  # [B, 3+D, npoint, nsample]

        # MLP on grouped points
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, -1)[0]  # [B, D', npoint] or [B, D', 1]

        if self.group_all:
            new_xyz = torch.zeros(xyz.size(0), 3, 1, device=xyz.device)
        else:
            new_xyz = gather_operation(xyz, fps_idx)  # [B, 3, npoint]

        return new_xyz, new_points


class PointNetEncoder(nn.Module):
    """PointNet++ Encoder: 3-level hierarchy -> global feature"""
    def __init__(self, in_feature, hidden_dim=64):
        super().__init__()
        self.in_channel = in_feature
        hc = hidden_dim

        self.sa1 = PointNetSetAbstraction(256, 0.1, 32, self.in_channel, [hc, hc * 2])
        self.sa2 = PointNetSetAbstraction(128, 0.15, 64, hc * 2 + 3, [hc * 2, hc * 4])
        self.sa3 = PointNetSetAbstraction(None, None, None, hc * 4 + 3, [hc * 4, hc * 8], group_all=True)
        self.out_dim = hc * 8

    def forward(self, xyz, features=None):
        """
        Input:  xyz [B, 3, N], features [B, D, N] or None
        Output: global feature [B, out_dim]
        """
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        return l3_points.squeeze(-1)


#################################################################################
#                           1D Conv Temporal Blocks                             #
#################################################################################

class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    """Residual 1D Conv block with pre-activation pattern"""
    def __init__(self, n_in, n_state, dilation=1, activation='gelu', norm='GN', dropout=0.1):
        super().__init__()
        padding = dilation
        self.norm_type = norm

        # Normalization layers
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # Activation layers
        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "silu" or activation == "swish":
            self.activation1 = Swish()
            self.activation2 = Swish()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
        else:
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_orig = x

        # Pre-activation pattern: norm -> activation -> conv
        if self.norm_type == "LN":
            x = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
        else:
            x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv1(x)

        if self.norm_type == "LN":
            x = self.norm2(x.transpose(-2, -1)).transpose(-2, -1)
        else:
            x = self.norm2(x)
        x = self.activation2(x)
        x = self.conv2(x)
        x = self.dropout(x)

        return x + x_orig


class Resnet1D(nn.Module):
    """Stack of residual 1D conv blocks with growing dilation"""
    def __init__(self, n_in, n_depth, dilation_growth_rate=3, reverse_dilation=False,
                 activation='gelu', norm='GN', dropout=0.1):
        super().__init__()
        blocks = [
            ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth,
                          activation=activation, norm=norm, dropout=dropout)
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class TemporalEncoder(nn.Module):
    """Temporal encoder with strided convolutions for downsampling"""
    def __init__(self, in_dim, out_dim, hidden_dim=512, n_layers=2, stride=2, n_depth=3,
                 dilation_growth_rate=3, activation='gelu', norm='GN'):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride * 2, stride // 2

        # Input projection
        blocks.append(nn.Conv1d(in_dim, hidden_dim, 3, 1, 1))
        blocks.append(nn.ReLU())

        # Downsampling layers
        for _ in range(n_layers):
            blocks.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, filter_t, stride, pad_t),
                Resnet1D(hidden_dim, n_depth, dilation_growth_rate,
                        reverse_dilation=False, activation=activation, norm=norm),
            ))

        # Output projection
        blocks.append(nn.Conv1d(hidden_dim, out_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        """x: [B, C, T] -> [B, out_dim, T']"""
        return self.model(x)


class TemporalDecoder(nn.Module):
    """Temporal decoder with upsampling"""
    def __init__(self, in_dim, out_dim, hidden_dim=512, n_layers=2, stride=2, n_depth=3,
                 dilation_growth_rate=3, activation='gelu', norm='GN'):
        super().__init__()
        blocks = []

        # Input projection
        blocks.append(nn.Conv1d(in_dim, hidden_dim, 3, 1, 1))
        blocks.append(nn.ReLU())

        # Upsampling layers (with reverse dilation)
        for _ in range(n_layers):
            blocks.append(nn.Sequential(
                Resnet1D(hidden_dim, n_depth, dilation_growth_rate,
                        reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=stride, mode='linear', align_corners=False),
                nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1),
            ))

        # Output projection (no activation at the end for unconstrained output)
        blocks.append(nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(hidden_dim, out_dim, 3, 1, 1))

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        """x: [B, C, T'] -> [B, out_dim, T]"""
        return self.model(x)


#################################################################################
#                    Object-Centric Hand-Object Motion VAE                      #
#################################################################################
class ObjectCentricMotionAE(nn.Module):
    """
    Object-Centric VAE Architecture: Object is encoded and decoded independently.
    Hand decoder is conditioned on object latent.
    This ensures object pose is predicted ONCE, not averaged from multiple hands.
    
    VAE features:
    - Separate mu/logvar projections for object and hand latents
    - Reparameterization trick for training
    - KL divergence regularization
    """
    def __init__(self, cfg):
        super().__init__()
        self.latent_dim = getattr(cfg, 'latent_dim', 256)
        self.hidden_dim = getattr(cfg, 'hidden_dim', 512)
        self.pointnet_dim = getattr(cfg, 'pointnet_dim', 64)
        self.n_layers = getattr(cfg, 'n_layers', 2)
        self.stride = getattr(cfg, 'stride', 2)
        self.n_depth = getattr(cfg, 'n_depth', 3)
        
        self.pointnet = PointNetEncoder(in_feature=57, hidden_dim=self.pointnet_dim)
        self.spatial_dim = self.pointnet.out_dim
        
        self.obj_pose_feat_dim = 9
        self.obj_projector = nn.Sequential(nn.Linear(self.spatial_dim + self.obj_pose_feat_dim, self.latent_dim), nn.ReLU(), nn.Linear(self.latent_dim, self.latent_dim))
        self.obj_temporal_encoder = TemporalEncoder(in_dim=self.latent_dim, out_dim=self.latent_dim, hidden_dim=self.hidden_dim, n_layers=self.n_layers, stride=self.stride, n_depth=self.n_depth)
        self.obj_mu = nn.Conv1d(self.latent_dim, self.latent_dim, 1)
        self.obj_logvar = nn.Conv1d(self.latent_dim, self.latent_dim, 1)
        self.obj_output_dim = 9
        self.obj_temporal_decoder = TemporalDecoder(in_dim=self.latent_dim, out_dim=self.obj_output_dim, hidden_dim=self.hidden_dim, n_layers=self.n_layers, stride=self.stride, n_depth=self.n_depth)
        
        self.hand_pose_feat_dim = 96 + 3 + 3 + 96 + 48
        self.hand_projector = nn.Sequential(nn.Linear(self.spatial_dim + self.hand_pose_feat_dim, self.latent_dim), nn.ReLU(), nn.Linear(self.latent_dim, self.latent_dim))
        self.hand_temporal_encoder = TemporalEncoder(in_dim=self.latent_dim, out_dim=self.latent_dim, hidden_dim=self.hidden_dim, n_layers=self.n_layers, stride=self.stride, n_depth=self.n_depth)
        self.hand_mu = nn.Conv1d(self.latent_dim, self.latent_dim, 1)
        self.hand_logvar = nn.Conv1d(self.latent_dim, self.latent_dim, 1)
        self.hand_output_dim = 16 * 6 + 3
        self.hand_temporal_decoder = TemporalDecoder(in_dim=self.latent_dim * 2, out_dim=self.hand_output_dim, hidden_dim=self.hidden_dim, n_layers=self.n_layers, stride=self.stride, n_depth=self.n_depth)

    @staticmethod
    def fast_inverse_4x4(T):
        R = T[..., :3, :3]
        t = T[..., :3, 3:4]
        R_inv = R.transpose(-1, -2)
        t_inv = -torch.matmul(R_inv, t)
        T_inv = torch.zeros_like(T)
        T_inv[..., :3, :3] = R_inv
        T_inv[..., :3, 3:4] = t_inv
        T_inv[..., 3, 3] = 1.0
        return T_inv

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * eps
        Args:
            mu: [B, D, T'] mean
            logvar: [B, D, T'] log variance
        Returns:
            z: [B, D, T'] sampled latent
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_kine_feats(self, inputs):
        """Compute hand features with hand-object relative geometry"""
        batch_size, seq_len, _, _ = inputs["obj_transf"].shape
        
        # Transform object point cloud
        transf_obj_pcds = torch.einsum("btij,bnj->btni",  inputs['obj_transf'][:, :, :3, :3],  inputs['obj_pcds']) + inputs['obj_transf'][:, :, :3, 3].unsqueeze(2)
        
        # Inverse hand transforms
        inv_rh_transf = self.fast_inverse_4x4(inputs['rh_transf']).reshape((batch_size, seq_len, 16, 4, 4))
        
        # Object in hand-relative coordinates
        rh_pcds = dehomoify(torch.einsum("btpij,btnj->btpni", inv_rh_transf, homoify(transf_obj_pcds - inputs['rh_trans'].unsqueeze(2))))
        rh_pcds = rh_pcds.permute(0, 1, 3, 2, 4).reshape((batch_size, seq_len, -1, 48))
        
        # Root-relative object
        rw_pcds = transf_obj_pcds - inputs['rh_trans'].unsqueeze(2)
        
        # Kinematic point cloud features
        kine_pcds = torch.cat([inputs['obj_pcds'].unsqueeze(1).expand_as(transf_obj_pcds), transf_obj_pcds, rw_pcds, rh_pcds], dim=-1)
        
        # Hand pose features
        rh_poses_6d = inputs['rh_poses'][:, :, :, :2, :3].reshape(batch_size, seq_len, -1)
        rh_trans = inputs['rh_trans']
        
        # Relative object transform in hand coordinates
        rw_obj_transf = inputs["obj_transf"].clone()
        rw_obj_transf[:, :, :3, 3] = rw_obj_transf[:, :, :3, 3] - inputs['rh_trans']
        rw_obj_trans = rw_obj_transf[:, :, :3, 3]
        
        rh_obj_transf = torch.einsum("btpij,btjk->btpik", inv_rh_transf, rw_obj_transf)
        rh_obj_trans = rh_obj_transf[:, :, :, :3, 3].reshape(batch_size, seq_len, -1)
        rh_obj_rot_6d = rh_obj_transf[:, :, :, :2, :3].reshape(batch_size, seq_len, -1)
        
        hand_pose_feat = torch.cat([rh_poses_6d, rh_trans, rw_obj_trans, rh_obj_rot_6d, rh_obj_trans], dim=-1)

        obj_rot_6d = inputs['obj_transf'][:, :, :2, :3].reshape(batch_size, seq_len, -1)
        obj_trans = inputs['obj_transf'][:, :, :3, 3]
        obj_pose_feat = torch.cat([obj_rot_6d, obj_trans], dim=-1)
        
        return kine_pcds, hand_pose_feat, obj_pose_feat

    def encode_object(self, spatial_feat, obj_pose_feat, sample=True):
        """
        Encode object motion from pre-computed features.

        Args:
            spatial_feat: [B*T, spatial_dim] shared PointNet output
            obj_pose_feat: [B, T, 9] object pose features (rot_6d + trans)
            sample: if True, return sampled z; if False, return mu only

        Returns:
            z_obj, obj_mu, obj_logvar (if sample=True) or obj_mu (if sample=False)
        """
        batch_size, seq_len = obj_pose_feat.shape[:2]

        obj_pose_feat_flat = obj_pose_feat.reshape(batch_size * seq_len, -1)
        obj_combined_feat = torch.cat([spatial_feat, obj_pose_feat_flat], dim=-1)

        obj_latent = self.obj_projector(obj_combined_feat)
        obj_latent = obj_latent.reshape(batch_size, seq_len, -1)

        h_obj = self.obj_temporal_encoder(obj_latent.permute(0, 2, 1))  # [B, D, T']

        obj_mu = self.obj_mu(h_obj)
        obj_logvar = self.obj_logvar(h_obj)

        if sample:
            z_obj = self.reparameterize(obj_mu, obj_logvar)
            return z_obj, obj_mu, obj_logvar
        else:
            return obj_mu

    def encode_hand(self, spatial_feat, hand_pose_feat, sample=True):
        """
        Encode hand motion from pre-computed features.

        Args:
            spatial_feat: [B*T, spatial_dim] shared PointNet output
            hand_pose_feat: [B, T, 246] hand pose features
            sample: if True, return sampled z; if False, return mu only

        Returns:
            z_hand, hand_mu, hand_logvar (if sample=True) or hand_mu (if sample=False)
        """
        batch_size, seq_len = hand_pose_feat.shape[:2]

        hand_pose_feat_flat = hand_pose_feat.reshape(batch_size * seq_len, -1)
        hand_combined_feat = torch.cat([spatial_feat, hand_pose_feat_flat], dim=-1)

        hand_latent = self.hand_projector(hand_combined_feat)
        hand_latent = hand_latent.reshape(batch_size, seq_len, -1)

        h_hand = self.hand_temporal_encoder(hand_latent.permute(0, 2, 1))  # [B, D, T']

        hand_mu = self.hand_mu(h_hand)
        hand_logvar = self.hand_logvar(h_hand)

        if sample:
            z_hand = self.reparameterize(hand_mu, hand_logvar)
            return z_hand, hand_mu, hand_logvar
        else:
            return hand_mu

    def decode_object(self, z_obj):
        """Decode object pose from object latent"""
        obj_out = self.obj_temporal_decoder(z_obj)
        return obj_out.permute(0, 2, 1)

    def decode_hand(self, z_hand, z_obj):
        """Decode hand pose conditioned on object latent"""
        z_combined = torch.cat([z_hand, z_obj.detach()], dim=1)
        hand_out = self.hand_temporal_decoder(z_combined)
        return hand_out.permute(0, 2, 1)

    def encode(self, inputs, sample=True):
        """
        Encode both object and hand
        
        Args:
            inputs: dict with all required data
            sample: if True, return sampled z with mu/logvar; if False, return mu only
        
        Returns:
            If sample=True:
                z_obj, obj_mu, obj_logvar, z_hand, hand_mu, hand_logvar
            If sample=False:
                z_obj, z_hand (both are mu values)
        """
        batch_size = inputs["obj_pcds"].shape[0]
        num_points = inputs["obj_pcds"].shape[1]
        seq_len = inputs["obj_transf"].shape[1]
        kine_pcds, hand_pose_feat, obj_pose_feat = self.compute_kine_feats(inputs)
        kine_pcds = kine_pcds.reshape(batch_size * seq_len, num_points, -1).permute(0, 2, 1)
        xyz = kine_pcds[:, :3, :].contiguous()
        features = kine_pcds[:, 3:, :].contiguous()
        spatial_feat = self.pointnet(xyz, features)

        if sample:
            z_obj, obj_mu, obj_logvar = self.encode_object(spatial_feat, obj_pose_feat, sample=True)
            z_hand, hand_mu, hand_logvar = self.encode_hand(spatial_feat, hand_pose_feat, sample=True)
            return z_obj, obj_mu, obj_logvar, z_hand, hand_mu, hand_logvar
        else:
            z_obj = self.encode_object(spatial_feat, obj_pose_feat, sample=False)
            z_hand = self.encode_hand(spatial_feat, hand_pose_feat, sample=False)
            return z_obj, z_hand

    def decode(self, z_obj, z_hand):
        """Decode both object and hand"""
        rec_obj = self.decode_object(z_obj)
        rec_hand = self.decode_hand(z_hand, z_obj)
        return rec_obj, rec_hand

    def forward(self, inputs, sample=True, **kwargs):
        """
        Forward pass with VAE encoding and decoding
        
        Args:
            inputs: dict with all required data
            sample: if True, use reparameterization; if False, use mean (for inference)
        
        Returns:
            outputs dict with:
                - rec_poses, rec_hand, rec_obj: reconstructions
                - z_obj, z_hand: latent samples (or means if sample=False)
                - obj_mu, obj_logvar, hand_mu, hand_logvar: VAE parameters (only if sample=True)
        """
        outputs = {}
        
        # Encode
        if sample:
            z_obj, obj_mu, obj_logvar, z_hand, hand_mu, hand_logvar = self.encode(inputs, sample=True)
            outputs['obj_mu'] = obj_mu
            outputs['obj_logvar'] = obj_logvar
            outputs['hand_mu'] = hand_mu
            outputs['hand_logvar'] = hand_logvar
        else:
            z_obj, z_hand = self.encode(inputs, sample=False)
        
        # Decode
        rec_obj, rec_hand = self.decode(z_obj, z_hand)
        
        outputs['rec_poses'] = torch.cat([rec_hand, rec_obj], dim=-1)
        outputs['rec_hand'] = rec_hand
        outputs['rec_obj'] = rec_obj
        outputs['z_obj'] = z_obj
        outputs['z_hand'] = z_hand
        
        return outputs