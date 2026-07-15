"""
Modelo híbrido: forecasting (W-LSTMix original, congelável/finetunável)
+ cabeça de classificação por timestep (sua contribuição).

Arquitetura:
  * backbone = models.W_LSTMix.Model do repo original (inalterado):
      (trend_input, season_input) -> (trend_pred, season_pred)
  * head = MLP sobre a concatenação de:
      [trend_norm, season_norm] do backcast  +  [trend_pred, season_pred]
    produzindo um logit POR TIMESTEP do horizonte de forecast.
    Intuição: a previsão do backbone funciona como "modelo do normal";
    o head aprende a mapear (contexto, previsão) -> probabilidade de anomalia.
"""
from __future__ import annotations
import sys
import torch
import torch.nn as nn
from config import CFG

sys.path.append("./models")            # aponte para a pasta models do W-LSTMix
from models import W_LSTMix            # noqa: E402  (import do repo original)


class HybridWLSTMix(nn.Module):
    def __init__(self, device, freeze_backbone: bool = False,
                 pretrained_path: str | None = None):
        super().__init__()
        self.backbone = W_LSTMix.Model(
            device=device,
            num_blocks_per_stack=CFG.num_blocks_per_stack,
            forecast_length=CFG.forecast_length,
            backcast_length=CFG.backcast_length,
            patch_size=CFG.patch_size,
            num_patches=CFG.backcast_length // CFG.patch_size,
            thetas_dim=CFG.thetas_dim,
            hidden_dim=CFG.hidden_dim,
            embed_dim=CFG.embed_dim,
            num_heads=CFG.num_heads,
            ff_hidden_dim=CFG.ff_hidden_dim,
        )
        if pretrained_path:
            # LINHA CRUCIAL (fine-tuning): carrega os pesos publicados do
            # W-LSTMix antes de acoplar a cabeça de classificação.
            self.backbone.load_state_dict(
                torch.load(pretrained_path, map_location=device), strict=False)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        in_dim = 2 * CFG.backcast_length + 2 * CFG.forecast_length
        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(),
            # LINHA CRUCIAL: um logit por passo do horizonte -> detecção
            # de anomalia PONTUAL (por timestep), não por janela inteira.
            nn.Linear(128, CFG.forecast_length),
        )

    def forward(self, trend_in, season_in):
        trend_pred, season_pred = self.backbone(trend_in, season_in)
        feats = torch.cat([trend_in, season_in, trend_pred, season_pred], dim=-1)
        logits = self.cls_head(feats)              # (B, forecast_length)
        return trend_pred, season_pred, logits
