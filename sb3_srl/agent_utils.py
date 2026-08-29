#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:48:04 2025

@author: angel
"""
import json

from stable_baselines3.common.utils import get_latest_run_id


def parse_training_args(parser,
                        steps=50000,
                        memory_steps=5000,
                        batch_size=512,
                        eval_interval=10000,
                        eval_steps=60):
    arg_training = parser.add_argument_group('Training')
    arg_training.add_argument("--steps", type=int, default=steps,  # 25h at 25 frames
                              help='Number of training steps.')
    arg_training.add_argument('--memory-steps', type=int, default=memory_steps,
                              help='Number of steps for initial population of the Experience replay buffer.')
    arg_training.add_argument("--batch-size", type=int, default=batch_size,
                              help='Minibatch size for training.')
    arg_training.add_argument('--eval-interval', type=int, default=eval_interval,  # 30m at 25 frames
                              help='Steps interval for progress evaluation.')
    arg_training.add_argument('--eval-steps', type=int, default=eval_steps,  # 1m at 25 frames
                              help='Number of evaluation steps.')
    return arg_training


def parse_memory_args(parser):
    arg_mem = parser.add_argument_group('Memory buffer')
    arg_mem.add_argument("--memory-capacity", type=int, default=65536,  # 2**16
                           help='Maximum number of transitions in the Experience replay buffer.')
    arg_mem.add_argument("--memory-prioritized", action='store_true',
                           help='Whether if memory buffer is Prioritized experiencie replay or not.')
    arg_mem.add_argument("--prioritized-alpha", type=float, default=0.6,
                           help='Alpha prioritization exponent for PER.')
    arg_mem.add_argument("--prioritized-initial-beta", type=float, default=0.4,
                           help='Beta bias for sampling for PER.')
    arg_mem.add_argument("--beta-steps", type=float, default=112500,
                           help='Beta bias steps to reach 1.')
    return arg_mem


def parse_utils_args(parser):
    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--use-cuda', action='store_true',
                           help='Flag specifying whether to use the GPU.')
    arg_utils.add_argument('--seed', type=int, default=666,
                           help='Seed valu for torch and nummpy.')
    arg_utils.add_argument('--logspath', type=str, default=None,
                           help='Specific output path for training results.')
    return arg_utils


def parse_srl_args(parser):
    arg_srl = parser.add_argument_group(
        'State representation learning variation')
    arg_srl.add_argument("--is-srl", action='store_true',
                         help='Whether if method is SRL-based or not.')
    arg_srl.add_argument("--latent-dim", type=int, default=32,
                         help='Number of features in the latent representation Z.')
    arg_srl.add_argument("--feature-dim", type=int, default=32,
                         help='Number of features from the encoder.')
    arg_srl.add_argument("--hidden-dim", type=int, default=512,
                         help='Number of units in the hidden layers.')
    arg_srl.add_argument("--num-filters", type=int, default=32,
                         help='Number of filters in the CNN hidden layers.')
    arg_srl.add_argument("--num-layers", type=int, default=1,
                         help='Number of hidden layers.')
    arg_srl.add_argument("--encoder-lr", type=float, default=1e-3,
                         help='Encoder function Adam learning rate.')
    arg_srl.add_argument("--encoder-tau", type=float, default=0.999,
                         help='Encoder tau polyak update.')
    # arg_srl.add_argument("--encoder-steps", type=int, default=9000,
    #                      help='Steps of no improvement to stop Encoder gradient.')
    arg_srl.add_argument("--decoder-lr", type=float, default=1e-3,
                         help='Decoder function Adam learning rate.')
    arg_srl.add_argument("--decoder-latent-lambda", type=float, default=1e-6,
                         help='Decoder regularization lambda value.')
    # arg_srl.add_argument("--decoder-weight-decay", type=float, default=1e-7,
    #                      help='Decoder function Adam weight decay value.')
    arg_srl.add_argument("--representation-freq", type=int, default=1,
                         help='Steps interval for AE batch training.')
    arg_srl.add_argument("--encoder-only", action='store_true',
                         help='Whether if use the SRL loss.')
    arg_srl.add_argument("--joint-optimization", action='store_true',
                         help='Whether if jointly optimize representation with RL updates.')
    arg_srl.add_argument("--use-stochastic", action='store_true',
                         help='Whether if use the Stochastic version model.')

    arg_srl.add_argument("--model-reconstruction", action='store_true',
                         help='Whether if use the Reconstruction model.')
    arg_srl.add_argument("--model-reconstruction-dist", action='store_true',
                         help='Whether if use the ReconstructionDist reconstruction model.')
    arg_srl.add_argument("--model-spr", action='store_true',
                         help='Whether if use the SelfPredictive model.')
    arg_srl.add_argument("--model-ispr", action='store_true',
                         help='Whether if use the InfoNCE SimpleSPR version model.')
    arg_srl.add_argument("--model-proprio", action='store_true',
                         help='Whether if use the Proprioceptive version model.')
    # arg_srl.add_argument("--model-ispr-mumo", action='store_true',
    #                      help='Whether if use the InfoNCE SimpleSPR Multimodal version model.')
    # arg_srl.add_argument("--model-i2spr", action='store_true',
    #                      help='Whether if use the Introspective InfoNCE SimpleSPR model.')
    # arg_srl.add_argument("--introspection-lambda", type=float, default=0,
    #                      help='Introspection loss function lambda value, >0 to use introspection.')

    arg_srl.add_argument("--fusion-mlp", action='store_true',
                         help='Use MLP for Proprioceptive fusion.')
    arg_srl.add_argument("--fusion-conv1d", action='store_true',
                         help='Use 1D convolution for Proprioceptive fusion.')
    arg_srl.add_argument("--fusion-gated", action='store_true',
                         help='Use Gated Proprioceptive fusion.')
    arg_srl.add_argument("--fusion-film", action='store_true',
                         help='Use FiLM Proprioceptive fusion.')
    arg_srl.add_argument("--fusion-crossatt", action='store_true',
                         help='Use Attention-based Proprioceptive fusion.')
    # arg_srl.add_argument("--fusion-mamba", action='store_true',
    #                      help='Use Mamba model for Proprioceptive fusion.')
    arg_srl.add_argument("--late-fusion", action='store_true',
                         help='Process sensor fusion in downstream processes.')
    return arg_srl


def args2encoder(args, env_params):
    _args = args
    if not isinstance(_args, dict):
        _args = vars(_args)
    params = {
        'state_shape': env_params['state_shape'],
        'feature_dim': _args.get('feature_dim', 32),
        'latent_dim': _args.get('latent_dim', 32),
        'layers_dim': [_args.get('hidden_dim', 256)] * _args.get('num_layers', 2),
    }

    encoder = 'Vector'

    if _args.get('model_proprio', False):
        encoder = 'AdPu'
        params['prop_mask'] = env_params['prop_mask']
        params['pixel_shape'] = None
        params['pixel_dim'] = None

        #multimodal
        if _args.get('is_pixels', False) and _args.get('is_vector', False):
            params['state_shape'] = env_params['state_shape'][0]
            params['pixel_shape'] = env_params['state_shape'][1]
            params['pixel_dim'] = 50

    elif _args.get('model_spr', False):
        encoder = 'SimpleSPR'

    elif _args.get('is_pixels', False):
        encoder = 'NatureCNN'
        del params['layers_dim']
        params['is_pixels'] = True
        params['features_dim'] = 512
        params['normalized_image'] = False

    return encoder, params


def args2decoder(args, env_params):
    _args = args
    if not isinstance(_args, dict):
        _args = vars(_args)
    params = {
        'state_shape': env_params['state_shape'],
        'latent_dim': _args.get('latent_dim', 32),
        'layers_dim': [_args.get('hidden_dim', 256)] * _args.get('num_layers', 2),
    }

    decoder = 'Vector'

    if _args.get('model_proprio', False):
        decoder = 'ProprioceptiveSPR'
        params['action_shape'] = env_params['action_shape']
        params['with_fusion'] = False
        del params['state_shape']

    elif _args.get('model_spr', False):
        decoder = 'SPR'
        if _args.get('is_pixels', False):
            params['layers_dim'] = [params['layers_dim'][-1]] * (len(params['layers_dim']) - 1)
        params['action_shape'] = env_params['action_shape']
        del params['state_shape']

    elif _args.get('model_ispr', False):
        decoder = 'SimpleSPR'
        if _args.get('is_pixels', False):
            params['layers_dim'] = [params['layers_dim'][-1]] * (len(params['layers_dim']) - 1)
        params['action_shape'] = env_params['action_shape']
        del params['state_shape']

    elif _args.get('is_pixels', False):
        decoder = 'Pixel'
        params['is_pixels'] = True
        params['layers_dim'] = [params['layers_dim'][-1]] * (len(params['layers_dim']) - 1)

    return decoder, params


def args2pipeline(args, env_params):
    _args = args
    if not isinstance(_args, dict):
        _args = vars(_args)
        
    pipeline = {'representation': []}

    if not _args.get('model_proprio', False):
        return pipeline

    fusion, params = None, {}
    params['latent_dim'] = _args.get('latent_dim', 32)

    if _args.get('fusion_mlp', False):
        fusion = 'mlp'
    if _args.get('fusion_conv1d', False):
        fusion = 'conv1d'
    if _args.get('fusion_gated', False):
        fusion = 'gated'
    if _args.get('fusion_film', False):
        fusion = 'film'
    if _args.get('fusion_crossatt', False):
        fusion = 'crossatt'
    # if _args.get('fusion_mamba', False):
    #     fusion = 'mamba'

    if fusion is not None:
        fusion = "F:" + fusion
        pipeline['representation'].append((fusion, params))

    if _args.get('late_fusion', False):
        pipeline['critic'] = pipeline['representation'].copy()

    return pipeline


def args2srl_config(args, env_params):
    _args = args
    if not isinstance(_args, dict):
        _args = vars(_args)

    loss_name = None
    model_name = None

    encoder = args2encoder(args, env_params)

    loss_args = {
        'encoder_lr': _args.get('encoder_lr', 1e-3),
        'encoder_tau': _args.get('encoder_tau', 0.999),
        # 'encoder_steps': _args.get('encoder_steps', 9000),
    }

    decoder = None
    loss_args['decoder_lr'] = None
    if not _args.get('encoder_only', False):
        decoder = args2decoder(args, env_params)
        loss_args['decoder_lr'] = _args.get('decoder_lr', 1e-3)
        loss_args['decoder_lambda'] = _args.get('decoder_lambda', 1e-6)
        # loss_args['decoder_weight_decay'] = _args.get('decoder_weight_decay', 1e-7)

    if _args.get('model_reconstruction', False):
        loss_name = 'Reconstruction'
        model_name = loss_name

    elif _args.get('model_reconstruction_dist', False):
        loss_name = 'ReconstructionDist'
        model_name = loss_name

    elif _args.get('model_spr', False):
        loss_name = 'SelfPredictive'
        model_name = loss_name

    elif _args.get('model_ispr', False):
        loss_name = 'InfoSPR'
        model_name = loss_name

    # elif _args.get('model_i2spr', False):
    #     loss_name = 'IntrospectiveInfoSPR'

    elif _args.get('model_proprio', False):
        loss_name = 'InfoSPR'
        model_name = 'Proprioceptive'

    else:
        raise ValueError('SRL model not recognized...')

    pipeline = args2pipeline(args, env_params)

    srl_config = {
        'encoder': encoder,
        'loss': (loss_name, loss_args),
        'pipeline': pipeline,
        'decoder': decoder,
        'is_stochastic': _args.get('use_stochastic', False),
        'joint_optimization': _args.get('joint_optimization', False),
    }

    print(srl_config)

    return {'model': model_name, 'config': srl_config}


def args2logpath(args, algo, env_name=None):
    if args.logspath is not None:
        return args.logspath

    # Summary folder
    outfolder = "logs"
    if env_name is not None:
        outfolder += f"/{env_name}/"

    path_suffix = ''
    # method labels
    if args.model_reconstruction:
        path_suffix += '-rec'
    if args.model_reconstruction_dist:
        path_suffix += '-drec'
    if args.model_spr:
        path_suffix += '-spr'
    if args.model_ispr:
        path_suffix += '-ispr'
    # if args.model_i2spr:
    #     path_suffix += '-i2spr'
    # if args.model_ispr_mumo:
    #     path_suffix += '-ispr-custom'
    if args.model_proprio:
        path_suffix += '-proprio'
    # extra labels
    # if args.introspection_lambda != 0.:
    #     path_suffix += '-intr'
    if args.joint_optimization:
        path_suffix += '-joint'
    if args.use_stochastic:
        path_suffix += '-stch'
    # fusion labels
    if args.fusion_mlp:
        path_suffix += '-fmlp'
    if args.fusion_conv1d:
        path_suffix += '-fconv1d'
    if args.fusion_gated:
        path_suffix += '-fgated'
    if args.fusion_film:
        path_suffix += '-ffilm'
    if args.fusion_crossatt:
        path_suffix += '-fcrossatt'

    if args.late_fusion:
        path_suffix += '_late'

    exp_name = f"{algo}{path_suffix}"

    latest_run_id = get_latest_run_id(outfolder, exp_name)

    return outfolder, exp_name, latest_run_id


def save_dict_json(dict2save, json_path):
    proc_dic = dict2save.copy()
    dict_json = json.dumps(proc_dic,
                           indent=4,
                           default=lambda o: str(o))
    with open(json_path, 'w') as jfile:
        jfile.write(dict_json)
    return dict_json


def load_json_dict(json_path):
    json_dict = dict()
    with open(json_path, 'r') as jfile:
        json_dict = json.load(jfile)
    return json_dict
