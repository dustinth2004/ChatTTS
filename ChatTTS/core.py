
import os
import logging
from omegaconf import OmegaConf

import torch
from vocos import Vocos
from .model.dvae import DVAE
from .model.gpt import GPT_warpper
from .utils.gpu_utils import select_device
from .utils.io_utils import get_latest_modified_file
from .infer.api import refine_text, infer_code

from huggingface_hub import snapshot_download

logging.basicConfig(level = logging.INFO)


class Chat:
    """A chat class for synthesizing speech from text.

    This class provides a high-level interface for loading pre-trained models
    and generating audio from text inputs. It supports both English and Chinese
    languages, with options for fine-grained control over prosodic features.

    Attributes:
        pretrain_models (dict): A dictionary to store the pre-trained models.
        logger (logging.Logger): A logger for recording events and debugging.
    """
    def __init__(self, ):
        """Initializes the Chat class."""
        self.pretrain_models = {}
        self.logger = logging.getLogger(__name__)
        
    def check_model(self, level = logging.INFO, use_decoder = False):
        """Checks if all required models are initialized.

        Args:
            level (int, optional): The logging level for messages.
                Defaults to logging.INFO.
            use_decoder (bool, optional): Whether to check for the decoder model.
                Defaults to False.

        Returns:
            bool: True if all required models are initialized, False otherwise.
        """
        not_finish = False
        check_list = ['vocos', 'gpt', 'tokenizer']
        
        if use_decoder:
            check_list.append('decoder')
        else:
            check_list.append('dvae')
            
        for module in check_list:
            if module not in self.pretrain_models:
                self.logger.log(logging.WARNING, f'{module} not initialized.')
                not_finish = True
                
        if not not_finish:
            self.logger.log(level, f'All initialized.')
            
        return not not_finish
        
    def load_models(self, source='huggingface', force_redownload=False, local_path='<LOCAL_PATH>'):
        """Loads the pre-trained models from the specified source.

        Args:
            source (str, optional): The source to load the models from.
                Can be 'huggingface' or 'local'. Defaults to 'huggingface'.
            force_redownload (bool, optional): Whether to force redownload of
                the models from Hugging Face. Defaults to False.
            local_path (str, optional): The local path to the models.
                Required if source is 'local'. Defaults to '<LOCAL_PATH>'.
        """
        if source == 'huggingface':
            hf_home = os.getenv('HF_HOME', os.path.expanduser("~/.cache/huggingface"))
            try:
                download_path = get_latest_modified_file(os.path.join(hf_home, 'hub/models--2Noise--ChatTTS/snapshots'))
            except:
                download_path = None
            if download_path is None or force_redownload: 
                self.logger.log(logging.INFO, f'Download from HF: https://huggingface.co/2Noise/ChatTTS')
                download_path = snapshot_download(repo_id="2Noise/ChatTTS", allow_patterns=["*.pt", "*.yaml"])
            else:
                self.logger.log(logging.INFO, f'Load from cache: {download_path}')
            self._load(**{k: os.path.join(download_path, v) for k, v in OmegaConf.load(os.path.join(download_path, 'config', 'path.yaml')).items()})
        elif source == 'local':
            self.logger.log(logging.INFO, f'Load from local: {local_path}')
            self._load(**{k: os.path.join(local_path, v) for k, v in OmegaConf.load(os.path.join(local_path, 'config', 'path.yaml')).items()})
        
    def _load(
        self, 
        vocos_config_path: str = None, 
        vocos_ckpt_path: str = None,
        dvae_config_path: str = None,
        dvae_ckpt_path: str = None,
        gpt_config_path: str = None,
        gpt_ckpt_path: str = None,
        decoder_config_path: str = None,
        decoder_ckpt_path: str = None,
        tokenizer_path: str = None,
        device: str = None
    ):
        """Loads the individual pre-trained models.

        Args:
            vocos_config_path (str, optional): Path to the Vocos
                configuration file. Defaults to None.
            vocos_ckpt_path (str, optional): Path to the Vocos checkpoint file.
                Defaults to None.
            dvae_config_path (str, optional): Path to the DVAE configuration
                file. Defaults to None.
            dvae_ckpt_path (str, optional): Path to the DVAE checkpoint file.
                Defaults to None.
            gpt_config_path (str, optional): Path to the GPT configuration
                file. Defaults to None.
            gpt_ckpt_path (str, optional): Path to the GPT checkpoint file.
                Defaults to None.
            decoder_config_path (str, optional): Path to the decoder
                configuration file. Defaults to None.
            decoder_ckpt_path (str, optional): Path to the decoder checkpoint
                file. Defaults to None.
            tokenizer_path (str, optional): Path to the tokenizer file.
                Defaults to None.
            device (str, optional): The device to load the models on. If None,
                it will be automatically selected. Defaults to None.
        """
        if not device:
            device = select_device(4096)
            self.logger.log(logging.INFO, f'use {device}')
            
        if vocos_config_path:
            vocos = Vocos.from_hparams(vocos_config_path).to(device).eval()
            assert vocos_ckpt_path, 'vocos_ckpt_path should not be None'
            vocos.load_state_dict(torch.load(vocos_ckpt_path))
            self.pretrain_models['vocos'] = vocos
            self.logger.log(logging.INFO, 'vocos loaded.')
        
        if dvae_config_path:
            cfg = OmegaConf.load(dvae_config_path)
            dvae = DVAE(**cfg).to(device).eval()
            assert dvae_ckpt_path, 'dvae_ckpt_path should not be None'
            dvae.load_state_dict(torch.load(dvae_ckpt_path, map_location='cpu'))
            self.pretrain_models['dvae'] = dvae
            self.logger.log(logging.INFO, 'dvae loaded.')
            
        if gpt_config_path:
            cfg = OmegaConf.load(gpt_config_path)
            gpt = GPT_warpper(**cfg).to(device).eval()
            assert gpt_ckpt_path, 'gpt_ckpt_path should not be None'
            gpt.load_state_dict(torch.load(gpt_ckpt_path, map_location='cpu'))
            self.pretrain_models['gpt'] = gpt
            spk_stat_path = os.path.join(os.path.dirname(gpt_ckpt_path), 'spk_stat.pt')
            assert os.path.exists(spk_stat_path), f'Missing spk_stat.pt: {spk_stat_path}'
            self.pretrain_models['spk_stat'] = torch.load(spk_stat_path).to(device)
            self.logger.log(logging.INFO, 'gpt loaded.')
            
        if decoder_config_path:
            cfg = OmegaConf.load(decoder_config_path)
            decoder = DVAE(**cfg).to(device).eval()
            assert decoder_ckpt_path, 'decoder_ckpt_path should not be None'
            decoder.load_state_dict(torch.load(decoder_ckpt_path, map_location='cpu'))
            self.pretrain_models['decoder'] = decoder
            self.logger.log(logging.INFO, 'decoder loaded.')
        
        if tokenizer_path:
            tokenizer = torch.load(tokenizer_path, map_location='cpu')
            tokenizer.padding_side = 'left'
            self.pretrain_models['tokenizer'] = tokenizer
            self.logger.log(logging.INFO, 'tokenizer loaded.')
            
        self.check_model()
    
    def infer(
        self, 
        text, 
        skip_refine_text=False, 
        refine_text_only=False, 
        params_refine_text={}, 
        params_infer_code={}, 
        use_decoder=False
    ):
        """Synthesizes speech from text.

        Args:
            text (list of str): A list of text strings to synthesize.
            skip_refine_text (bool, optional): Whether to skip the text
                refinement step. Defaults to False.
            refine_text_only (bool, optional): Whether to only perform text
                refinement and return the refined text. Defaults to False.
            params_refine_text (dict, optional): Parameters for the text
                refinement step. Defaults to {}.
            params_infer_code (dict, optional): Parameters for the code
                inference step. Defaults to {}.
            use_decoder (bool, optional): Whether to use the decoder for
                generating audio. Defaults to False.

        Returns:
            list of numpy.ndarray: A list of synthesized audio waveforms as
                NumPy arrays.
        """
        
        assert self.check_model(use_decoder=use_decoder)
        
        if not skip_refine_text:
            text_tokens = refine_text(self.pretrain_models, text, **params_refine_text)['ids']
            text_tokens = [i[i < self.pretrain_models['tokenizer'].convert_tokens_to_ids('[break_0]')] for i in text_tokens]
            text = self.pretrain_models['tokenizer'].batch_decode(text_tokens)
            if refine_text_only:
                return text
            
        text = [params_infer_code.get('prompt', '') + i for i in text]
        params_infer_code.pop('prompt', '')
        result = infer_code(self.pretrain_models, text, **params_infer_code, return_hidden=use_decoder)
        
        if use_decoder:
            mel_spec = [self.pretrain_models['decoder'](i[None].permute(0,2,1)) for i in result['hiddens']]
        else:
            mel_spec = [self.pretrain_models['dvae'](i[None].permute(0,2,1)) for i in result['ids']]
            
        wav = [self.pretrain_models['vocos'].decode(i).cpu().numpy() for i in mel_spec]
        
        return wav
    
    def sample_random_speaker(self, ):
        """Samples a random speaker embedding.

        The speaker embedding is sampled from a Gaussian distribution with mean
        and standard deviation derived from the pre-trained speaker statistics.

        Returns:
            torch.Tensor: A random speaker embedding.
        """
        
        dim = self.pretrain_models['gpt'].gpt.layers[0].mlp.gate_proj.in_features
        std, mean = self.pretrain_models['spk_stat'].chunk(2)
        return torch.randn(dim, device=std.device) * std + mean
        

