C2_BD_MODELS = [
    'bact_phase_omni',
    'bact_fluor_omni',
    'worm_omni',
    'worm_bact_omni',
    'worm_high_res_omni',
    'cyto2_omni',
]

C2_MODELS = [
    'bact_phase_cp',
    'bact_fluor_cp',
    'plant_cp',  # 2D model for do_3D
    'worm_cp',
]

C1_BD_MODELS = ['plant_omni']

# This will be the affinity seg models
C1_MODELS = ['bact_phase_affinity']

CP_MODELS = ['cyto','nuclei','cyto2']
C2_MODEL_NAMES = C2_BD_MODELS + C2_MODELS + CP_MODELS
BD_MODEL_NAMES = C2_BD_MODELS + C1_BD_MODELS
MODEL_NAMES = C1_MODELS + C2_BD_MODELS + C1_BD_MODELS + C2_MODELS + CP_MODELS
