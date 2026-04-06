import json
import numpy as np
import torch

import train_dreamer_macro_edit as mod
from dreamer4.macro_edit import MacroDreamerEditModel, teacher_path_summary_dim


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    cache = '/Users/panziming/SwarmEcosystem/results/dreamer_macro_edit_v13_fixinit_qtrain_k4c384/segments.pt'
    payload = torch.load(cache, map_location='cpu', weights_only=False)
    train_samples = [mod.MacroSegmentSample(**item) for item in payload['train'][:16]]
    val_samples = [mod.MacroSegmentSample(**item) for item in payload['val'][:16]]

    model = MacroDreamerEditModel(
        max_vacancies=32,
        max_defects=64,
        max_shells=16,
        stats_dim=10,
        lattice_size=(40, 40, 40),
        neighbor_order='2NN',
        dim_latent=16,
        graph_hidden_size=32,
        patch_hidden_size=96,
        patch_latent_dim=64,
        path_latent_dim=32,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(4, include_stepwise_features=True),
        max_macro_k=16,
    )
    mod._initialize_output_heads(model, train_samples)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    train_loader = mod._build_loader(train_samples, batch_size=16, shuffle=False)
    val_loader = mod._build_loader(val_samples, batch_size=16, shuffle=False)
    weights = {
        'mask': 1.0,
        'type': 1.0,
        'pair': 0.0,
        'tau': 1.0,
        'reward': 0.5,
        'latent': 0.5,
        'proj': 0.5,
        'path': 0.05,
        'prior_edit': 0.25,
        'prior_latent': 0.25,
    }
    keys = [
        'tau_log_mae',
        'tau_scale_ratio',
        'change_f1',
        'change_topk_f1',
        'projected_change_f1',
        'changed_type_acc',
        'projected_changed_type_acc',
        'raw_vac_to_fe_count',
        'raw_fe_to_vac_count',
        'raw_matched_pair_count',
        'unchanged_vacancy_copy_acc',
    ]
    before = mod._evaluate(model, val_loader, 'cpu', max_changed_sites=8)
    train = mod._train_epoch(
        model,
        train_loader,
        optimizer,
        'cpu',
        max_changed_sites=8,
        weights=weights,
        epoch=1,
        total_epochs=1,
        tau_supervision_mode='prior_main',
    )
    after = mod._evaluate(model, val_loader, 'cpu', max_changed_sites=8)
    print(
        json.dumps(
            {
                'before': {key: round(float(before[key]), 4) for key in keys},
                'train': {
                    'loss': round(float(train['loss']), 4),
                    'mask': round(float(train['mask']), 4),
                    'type': round(float(train['type']), 4),
                    'prior_edit': round(float(train['prior_edit']), 4),
                    'vac_to_atom_type': round(float(train['vac_to_atom_type']), 4),
                    'atom_to_vac_type': round(float(train['atom_to_vac_type']), 4),
                },
                'after': {key: round(float(after[key]), 4) for key in keys},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
