#!/usr/bin/env python3

import argparse
import os
import gc
import json
import math
import multiprocessing
import shutil
import subprocess
import time
import numpy as np
import torch
import torch.nn as nn
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

# BioPython imports for PDB/CIF parsing
try:
    from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
    from Bio.SeqUtils import seq1
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: BioPython not available. PDB/CIF input will not be supported.")

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PRETRAINED_DIR = os.path.join(_PACKAGE_DIR, "pretrained")
DEFAULT_PPI_MODEL = os.path.join(_PRETRAINED_DIR, "bindscan.pt")

# Set multiprocessing start method to 'spawn' for proper CUDA isolation
# This must be done before any CUDA operations
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set


# ============================================================================
# PDB/CIF Processing Functions
# ============================================================================

def reindex_extractseq_pdbcif(input_file: str, output_file: str = None) -> Tuple[str, bool]:
    """
    Reindex residues in a PDB/CIF file to start from 1 and extract sequence.
    
    Args:
        input_file: Path to input PDB or CIF/mmCIF file
        output_file: Optional path to save reindexed structure
        
    Returns:
        Tuple of (sequence_string, already_formatted)
        - sequence_string: One-letter amino acid sequence
        - already_formatted: True if residues already start from 1
    """
    if not BIOPYTHON_AVAILABLE:
        raise RuntimeError("BioPython is required for PDB/CIF parsing. Install with: pip install biopython")
    
    ext = os.path.splitext(input_file)[1].lower()
    
    if ext in ['.cif', '.mmcif']:
        parser = MMCIFParser(QUIET=True)
    elif ext in ['.pdb', '.ent']:
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .pdb or .cif/.mmcif")
    
    structure = parser.get_structure('structure', input_file)
    chains = list(structure.get_chains())
    
    if len(chains) == 0:
        raise ValueError("No chains found in the structure file")
    if len(chains) > 1:
        chain_ids = [c.id for c in chains]
        raise ValueError(f"Structure contains multiple chains: {chain_ids}. Please provide a single-chain structure.")
    
    chain = chains[0]
    standard_residues = [r for r in chain.get_residues() if r.id[0] == ' ']
    
    if len(standard_residues) == 0:
        raise ValueError("No standard residues found in the structure")
    
    first_resid = standard_residues[0].id[1]
    already_formatted = (first_resid == 1)
    
    sequence = []
    for residue in standard_residues:
        resname = residue.resname
        try:
            one_letter = seq1(resname)
        except Exception:
            one_letter = 'X'
        sequence.append(one_letter)
    
    sequence_str = ''.join(sequence)
    
    if output_file:
        if already_formatted:
            shutil.copy(input_file, output_file)
        else:
            new_resid = 1
            for residue in standard_residues:
                residue.id = (' ', new_resid, ' ')
                new_resid += 1
            
            class SingleChainSelect(Select):
                def accept_residue(self, residue):
                    return residue.id[0] == ' '
            
            io = PDBIO()
            io.set_structure(structure)
            io.save(output_file, SingleChainSelect())
    
    return sequence_str, already_formatted


def _get_available_gpus() -> List[int]:
    """Get list of available GPU IDs without initializing CUDA."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return []


def _get_available_accelerators() -> Tuple[str, List[int]]:
    """
    Get available accelerators (CUDA GPUs or MPS).
    
    Returns:
        Tuple of (accelerator_type, device_list)
        - ('cuda', [0, 1, 2, ...]) for NVIDIA GPUs
        - ('mps', [0]) for Apple Silicon
        - ('cpu', [-1]) for CPU only
    """
    # Check for NVIDIA GPUs first
    gpus = _get_available_gpus()
    if gpus:
        return ('cuda', gpus)
    
    # Check for Apple MPS
    try:
        if torch.backends.mps.is_available():
            return ('mps', [0])  # MPS is single device
    except Exception:
        pass
    
    # CPU fallback
    return ('cpu', [-1])


# Imports - check availability
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Sequence generation will be skipped.")


# ============================================================================
# Utility Functions
# ============================================================================

def torch_gc():
    """Clear GPU/MPS memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Device: Apple MPS")
        return torch.device("mps")
    print("Device: CPU")
    return torch.device("cpu")


# Patch torch.load for newer PyTorch versions
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.pop('weights_only', None)
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load


# ============================================================================
# Sequence Generation Engine (LLM)
# ============================================================================

class HuggingfaceEngine:
    """
    HuggingFace-based sequence generation engine.
    """
    
    # Maximum target sequence length to include in prompt
    MAX_TARGET_SEQ_LENGTH = 1000
    
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.6,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 100,
        min_new_tokens: int = 30,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model - device_map="auto" works for CUDA, MPS, and CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model.eval()
        
        self._has_chat_template = (
            hasattr(self.tokenizer, 'chat_template') and 
            self.tokenizer.chat_template is not None
        )
    
    def _build_prompt(self, messages: list) -> str:
        """Build prompt from messages using chat template."""
        if self._has_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: Manual LLaMA 3/3.2 chat format
            prompt = "<|begin_of_text|>"
            for message in messages:
                role = message["role"]
                content = message["content"]
                prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt
    
    
    @torch.inference_mode()
    def batch_chat(self, messages: list, num_return_sequences: int = 8) -> List[str]:
        """
        Batch generation: Generate multiple sequences from the same prompt.
        
        This is much more efficient than calling chat() multiple times because:
        1. Single forward pass through the model
        2. Parallel token generation
        3. Better GPU utilization
        
        Args:
            messages: Chat messages (same format as chat())
            num_return_sequences: Number of sequences to generate (default: 8)
        
        Returns:
            List of generated sequences
        """
        prompt = self._build_prompt(messages)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Create generation config with num_return_sequences
        gen_config = GenerationConfig(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=True,
            num_beams=1,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=num_return_sequences,
            use_cache=True,  # Keep KV cache for faster generation
        )
        
        outputs = self.model.generate(
            **inputs,
            generation_config=gen_config,
        )
        
        input_length = inputs["input_ids"].shape[1]
        
        responses = []
        for i in range(num_return_sequences):
            generated_tokens = outputs[i][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)
        
        # Clean up tensors to prevent memory leak
        del inputs, outputs
        
        return responses


# ============================================================================
# QC System
# ============================================================================

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMVFW")
AROMATIC = set("FYW")
CHARGED = set("KRDE")
POSITIVE = set("KRH")
NEGATIVE = set("DE")

@dataclass
class QCConfig:
    """Configuration for QC filters."""
    min_length: int = 70
    max_length: int = 150
    max_single_aa_percent: float = 20.0
    min_hydrophobic_percent: float = 20.0
    max_hydrophobic_percent: float = 50.0
    min_aromatic_percent: float = 0.0
    max_aromatic_percent: float = 15.0
    min_charged_percent: float = 10.0
    max_charged_percent: float = 35.0
    min_entropy: float = 3.5
    max_repeat_length: int = 5
    max_net_charge: int = 10


@dataclass
class QCResult:
    """Results from QC analysis of a sequence."""
    sequence: str
    length: int
    is_valid_alphabet: bool
    invalid_chars: str
    max_single_aa: Optional[str]
    max_single_aa_percent: Optional[float]
    hydrophobic_percent: Optional[float]
    aromatic_percent: Optional[float]
    charged_percent: Optional[float]
    positive_percent: Optional[float]
    negative_percent: Optional[float]
    net_charge: Optional[int]
    shannon_entropy: Optional[float]
    longest_repeat: Optional[str]
    longest_repeat_length: Optional[int]
    has_homopolymer: Optional[bool]
    homopolymer_details: Optional[str]
    passed_all: bool = False
    failed_filters: List[str] = field(default_factory=list)


def check_valid_alphabet(sequence: str) -> Tuple[bool, str]:
    """Check if sequence contains only valid amino acids."""
    invalid = set(sequence) - VALID_AAS
    return len(invalid) == 0, ''.join(sorted(invalid))


def calculate_aa_composition(sequence: str) -> dict:
    """Calculate amino acid composition."""
    length = len(sequence)
    if length == 0:
        return {}
    
    counts = {}
    for aa in sequence:
        counts[aa] = counts.get(aa, 0) + 1
    
    return {aa: (count / length) * 100 for aa, count in counts.items()}


def get_max_single_aa(composition: dict) -> Tuple[str, float]:
    """Get the amino acid with highest percentage."""
    if not composition:
        return '', 0.0
    max_aa = max(composition, key=composition.get)
    return max_aa, composition[max_aa]


def calculate_group_percent(sequence: str, aa_group: set) -> float:
    """Calculate percentage of amino acids belonging to a group."""
    if len(sequence) == 0:
        return 0.0
    count = sum(1 for aa in sequence if aa in aa_group)
    return (count / len(sequence)) * 100


def calculate_net_charge(sequence: str) -> int:
    """Calculate net charge at pH 7."""
    positive = sum(1 for aa in sequence if aa in POSITIVE)
    negative = sum(1 for aa in sequence if aa in NEGATIVE)
    return positive - negative


def calculate_shannon_entropy(sequence: str) -> float:
    """Calculate Shannon entropy of sequence composition."""
    if len(sequence) == 0:
        return 0.0
    
    composition = calculate_aa_composition(sequence)
    entropy = 0.0
    
    for percent in composition.values():
        if percent > 0:
            p = percent / 100
            entropy -= p * math.log2(p)
    
    return entropy


def find_longest_repeat(sequence: str) -> Tuple[str, int]:
    """Find the longest repeated substring (tandem repeat)."""
    if len(sequence) < 2:
        return '', 0
    
    longest = ''
    
    for unit_len in range(1, len(sequence) // 2 + 1):
        for start in range(len(sequence) - unit_len):
            unit = sequence[start:start + unit_len]
            count = 1
            pos = start + unit_len
            
            while pos + unit_len <= len(sequence) and sequence[pos:pos + unit_len] == unit:
                count += 1
                pos += unit_len
            
            if count > 1:
                repeat = unit * count
                if len(repeat) > len(longest):
                    longest = repeat
    
    return longest, len(longest)


def find_homopolymers(sequence: str, min_length: int = 3) -> List[Tuple[str, int, int]]:
    """Find homopolymer runs (consecutive identical amino acids)."""
    homopolymers = []
    
    if len(sequence) == 0:
        return homopolymers
    
    current_aa = sequence[0]
    current_start = 0
    current_length = 1
    
    for i in range(1, len(sequence)):
        if sequence[i] == current_aa:
            current_length += 1
        else:
            if current_length >= min_length:
                homopolymers.append((current_aa, current_start, current_length))
            current_aa = sequence[i]
            current_start = i
            current_length = 1
    
    if current_length >= min_length:
        homopolymers.append((current_aa, current_start, current_length))
    
    return homopolymers


def qc1_prefilter(sequence: str, config: QCConfig) -> QCResult:
    """QC1: Basic pre-filter checks on original sequence.
    
    Only checks:
    - Valid amino acid alphabet
    - Length within range
    """
    # Basic checks only
    is_valid, invalid_chars = check_valid_alphabet(sequence)
    length = len(sequence)
    
    result = QCResult(
        sequence=sequence,
        length=length,
        is_valid_alphabet=is_valid,
        invalid_chars=invalid_chars,
        max_single_aa=None,
        max_single_aa_percent=None,
        hydrophobic_percent=None,
        aromatic_percent=None,
        charged_percent=None,
        positive_percent=None,
        negative_percent=None,
        net_charge=None,
        shannon_entropy=None,
        longest_repeat=None,
        longest_repeat_length=None,
        has_homopolymer=None,
        homopolymer_details=None,
    )
    
    # QC1: Only fail on alphabet and length
    failed = []
    if not is_valid:
        failed.append(f"invalid_alphabet({invalid_chars})")
    
    if length < config.min_length:
        failed.append(f"too_short({length}<{config.min_length})")
    elif length > config.max_length:
        failed.append(f"too_long({length}>{config.max_length})")
    
    result.failed_filters = failed
    result.passed_all = len(failed) == 0
    
    return result


def qc2_postfilter(sequence: str, config: QCConfig) -> QCResult:
    """QC2: Full quality checks on trimmed sequence.
    
    Checks all quality metrics:
    - Valid amino acid alphabet
    - Length within range  
    - Homopolymer runs
    - Charge balance
    - Shannon entropy
    - Amino acid composition (hydrophobic, aromatic, charged)
    - Tandem repeats
    """
    # Basic checks
    is_valid, invalid_chars = check_valid_alphabet(sequence)
    length = len(sequence)
    
    # Composition
    composition = calculate_aa_composition(sequence)
    max_aa, max_aa_percent = get_max_single_aa(composition)
    
    hydrophobic_pct = calculate_group_percent(sequence, HYDROPHOBIC)
    aromatic_pct = calculate_group_percent(sequence, AROMATIC)
    charged_pct = calculate_group_percent(sequence, CHARGED)
    positive_pct = calculate_group_percent(sequence, POSITIVE)
    negative_pct = calculate_group_percent(sequence, NEGATIVE)
    net_charge = calculate_net_charge(sequence)
    
    # Entropy and repeats
    entropy = calculate_shannon_entropy(sequence)
    longest_repeat, repeat_len = find_longest_repeat(sequence)
    homopolymers = find_homopolymers(sequence, min_length=config.max_repeat_length)
    
    homopolymer_details = '; '.join([f"{aa}x{l}@{pos}" for aa, pos, l in homopolymers])
    
    # Create result
    result = QCResult(
        sequence=sequence,
        length=length,
        is_valid_alphabet=is_valid,
        invalid_chars=invalid_chars,
        max_single_aa=max_aa,
        max_single_aa_percent=max_aa_percent,
        hydrophobic_percent=hydrophobic_pct,
        aromatic_percent=aromatic_pct,
        charged_percent=charged_pct,
        positive_percent=positive_pct,
        negative_percent=negative_pct,
        net_charge=net_charge,
        shannon_entropy=entropy,
        longest_repeat=longest_repeat,
        longest_repeat_length=repeat_len,
        has_homopolymer=len(homopolymers) > 0,
        homopolymer_details=homopolymer_details
    )
    
    # QC2: Apply ALL filters
    failed = []
    
    if not is_valid:
        failed.append(f"invalid_alphabet({invalid_chars})")
    
    if length < config.min_length:
        failed.append(f"too_short({length}<{config.min_length})")
    elif length > config.max_length:
        failed.append(f"too_long({length}>{config.max_length})")
    
    if max_aa_percent > config.max_single_aa_percent:
        failed.append(f"max_single_aa({max_aa}:{max_aa_percent:.1f}%>{config.max_single_aa_percent}%)")
    
    if hydrophobic_pct < config.min_hydrophobic_percent:
        failed.append(f"hydrophobic_low({hydrophobic_pct:.1f}%<{config.min_hydrophobic_percent}%)")
    elif hydrophobic_pct > config.max_hydrophobic_percent:
        failed.append(f"hydrophobic_high({hydrophobic_pct:.1f}%>{config.max_hydrophobic_percent}%)")
    
    if aromatic_pct > config.max_aromatic_percent:
        failed.append(f"aromatic_high({aromatic_pct:.1f}%>{config.max_aromatic_percent}%)")
    
    if charged_pct < config.min_charged_percent:
        failed.append(f"charged_low({charged_pct:.1f}%<{config.min_charged_percent}%)")
    elif charged_pct > config.max_charged_percent:
        failed.append(f"charged_high({charged_pct:.1f}%>{config.max_charged_percent}%)")
    
    if entropy < config.min_entropy:
        failed.append(f"low_entropy({entropy:.2f}<{config.min_entropy})")
    
    if repeat_len > config.max_repeat_length * 2:
        failed.append(f"long_repeat({longest_repeat})")
    
    if len(homopolymers) > 0:
        failed.append(f"homopolymer({homopolymer_details})")
    
    if abs(net_charge) > config.max_net_charge:
        failed.append(f"extreme_charge({net_charge})")
    
    result.failed_filters = failed
    result.passed_all = len(failed) == 0
    
    return result


# ============================================================================
# ESM2-8M Protein Embedding Model for PPI Prediction
# ============================================================================

MAX_SEQ_LEN = 1024
D_MODEL = 320  # ESM2-8M embedding dimension


class ESM2Wrapper:
    """Wrapper for ESM2-8M model to provide embeddings."""
    
    def __init__(self, device, max_len=1024):
        try:
            import esm
        except ImportError:
            raise ImportError("Please install fair-esm: pip install fair-esm")
        
        self.device = device
        self.max_len = max_len
        self.repr_layer = 6
        
        print("Loading ESM2-8M (esm2_t6_8M_UR50D)...")
        self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.model = self.model.to(device).eval()
        
        self.padding_idx = self.alphabet.padding_idx
        self.cls_idx = self.alphabet.cls_idx
        self.eos_idx = self.alphabet.eos_idx
        
        print(f"Loaded ESM2-8M: {D_MODEL}d × 6L × 20H")


class ESM2Tokenizer:
    """Tokenizer wrapper for ESM2."""
    
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.pad_id = alphabet.padding_idx
        self.cls_id = alphabet.cls_idx
        self.eos_id = alphabet.eos_idx
        self.mask_id = alphabet.mask_idx
    
    def encode(self, seq: str):
        """Encode protein sequence using ESM2 alphabet."""
        encoded = [self.cls_id]
        for aa in seq.upper():
            idx = self.alphabet.get_idx(aa)
            encoded.append(idx)
        encoded.append(self.eos_id)
        return encoded
    
    def __len__(self):
        return len(self.alphabet)


# ============================================================================
# PPI Classifier
# ============================================================================

"""
PPI_CONFIG = {
        'hidden': [256, 256, 512, 256, 128],
        'combine': 'all',
        'dropout': 0.1
        }
"""
class SiameseClassifier(nn.Module):
    """Siamese network for PPI classification."""
    
    def __init__(self, input_dim=320, hidden=[256, 256, 512, 256, 128], dropout=0.1, combine='symmetric'):
        super().__init__()
        self.combine_mode = combine
        
        # Split: first half encoder, rest classifier
        split = max(1, len(hidden) // 2)
        enc_dims, clf_dims = hidden[:split], hidden[split:]
        
        # Shared encoder
        layers = []
        prev = input_dim
        for d in enc_dims:
            layers += [nn.Linear(prev, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(dropout)]
            prev = d
        self.encoder = nn.Sequential(*layers)
        
        # Classifier input size based on combination mode
        enc_out = enc_dims[-1]
        clf_in = {'concat': 2, 'hadamard': 1, 'abs_diff': 1, 'all': 4, 'symmetric': 2}[combine] * enc_out
        
        # Classifier
        layers = []
        prev = clf_in
        for d in clf_dims:
            layers += [nn.Linear(prev, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(dropout)]
            prev = d
        layers.append(nn.Linear(prev, 1))
        self.classifier = nn.Sequential(*layers)
        
        self.enc_dims, self.clf_dims = enc_dims, clf_dims
    
    def forward(self, emb1, emb2):
        e1, e2 = self.encoder(emb1), self.encoder(emb2)
        
        if self.combine_mode == 'concat':
            x = torch.cat([e1, e2], dim=1)
        elif self.combine_mode == 'hadamard':
            x = e1 * e2
        elif self.combine_mode == 'abs_diff':
            x = torch.abs(e1 - e2)
        elif self.combine_mode == 'symmetric':
            # Order-invariant: f(A, B) == f(B, A)
            # Captures co-activation (e1*e2) and divergence (|e1-e2|)
            # as separate channels, no algebraic redundancy.
            x = torch.cat([e1 * e2, torch.abs(e1 - e2)], dim=1)
        elif self.combine_mode == 'all':
            x = torch.cat([e1, e2, e1 * e2, torch.abs(e1 - e2)], dim=1)
        else:
            raise ValueError(f"Unknown combine mode: {self.combine_mode}")
        
        return self.classifier(x).squeeze(-1)


class SiamesePPIPredictor:
    """Complete pipeline for PPI prediction using ESM2-8M embeddings."""
    
    def __init__(self, device=None, hidden=[256, 256, 512, 256, 128], 
                 combine='symmetric', dropout=0.1):
        self.device = device or get_device()
        self.hidden, self.combine, self.dropout = hidden, combine, dropout
        
        # Load ESM2-8M
        self.esm2_wrapper = ESM2Wrapper(self.device)
        self.tokenizer = ESM2Tokenizer(self.esm2_wrapper.alphabet)
        
        self.model = None
        
        # Precomputed target embedding - computed once via set_target()
        self._target_seq = None
        self._target_emb = None
    
    def _get_embedding(self, ids, mask):
        """Extract mean-pooled embeddings from ESM2-8M."""
        with torch.no_grad():
            results = self.esm2_wrapper.model(ids, repr_layers=[self.esm2_wrapper.repr_layer])
            embeddings = results["representations"][self.esm2_wrapper.repr_layer]
            
            # Mean pooling with mask (exclude padding, <cls>, <eos>)
            adjusted_mask = mask.clone().float()
            adjusted_mask[:, 0] = 0  # Exclude <cls>
            
            # Find and exclude <eos> for each sequence
            for j in range(ids.shape[0]):
                eos_positions = (ids[j] == self.esm2_wrapper.eos_idx).nonzero(as_tuple=False)
                if len(eos_positions) > 0:
                    adjusted_mask[j, eos_positions[0, 0]] = 0
            
            m = adjusted_mask.unsqueeze(-1)
            pooled = (embeddings * m).sum(1) / m.sum(1).clamp(min=1)
            
            # Explicitly delete intermediate tensors to prevent memory leak
            del results, embeddings, adjusted_mask, m
            
            return pooled
    
    def _init_model(self):
        self.model = SiameseClassifier(
            input_dim=D_MODEL, hidden=self.hidden, 
            dropout=self.dropout, combine=self.combine
        ).to(self.device)
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"Classifier: hidden={self.hidden}, combine={self.combine}, params={params:,}")
    
    def set_target(self, target_seq: str):
        """Precompute and cache the target embedding.
        
        Call this once after loading the model to avoid redundant computation
        during predict_batch() calls. The target embedding is computed once
        and reused for all subsequent predictions.
        
        Args:
            target_seq: Target protein sequence
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() before set_target().")
        
        max_aa = MAX_SEQ_LEN - 2
        target_key = target_seq[:max_aa]
        
        # Skip if already computed for this target
        if self._target_seq == target_key and self._target_emb is not None:
            return
        
        # Compute target embedding
        target_ids = self.tokenizer.encode(target_key)
        target_pad = MAX_SEQ_LEN - len(target_ids)
        target_ids_tensor = torch.tensor([target_ids + [self.tokenizer.pad_id] * target_pad], device=self.device)
        target_mask = torch.tensor([[1] * len(target_ids) + [0] * target_pad], device=self.device)
        
        with torch.no_grad():
            self._target_emb = self._get_embedding(target_ids_tensor, target_mask)
        
        self._target_seq = target_key
        print(f"Target embedding precomputed: {len(target_key)} residues")
    
    def load(self, path):
        """Load trained model."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.hidden, self.combine, self.dropout = ckpt['hidden'], ckpt['combine'], ckpt['dropout']
        self._init_model()
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        # Clear any existing target embedding (will need to call set_target again)
        self._target_seq = None
        self._target_emb = None
        print(f"Loaded: {path} (epoch {ckpt['epoch']}, loss {ckpt['val_loss']:.4f})")
    
    def predict(self, seq1, seq2):
        """Predict interaction probability for a single pair."""
        self.model.eval()
        max_aa = MAX_SEQ_LEN - 2
        
        ids1 = self.tokenizer.encode(seq1[:max_aa])
        ids2 = self.tokenizer.encode(seq2[:max_aa])
        pad1, pad2 = MAX_SEQ_LEN - len(ids1), MAX_SEQ_LEN - len(ids2)
        
        ids1_tensor = torch.tensor([ids1 + [self.tokenizer.pad_id] * pad1], device=self.device)
        ids2_tensor = torch.tensor([ids2 + [self.tokenizer.pad_id] * pad2], device=self.device)
        mask1 = torch.tensor([[1] * len(ids1) + [0] * pad1], device=self.device)
        mask2 = torch.tensor([[1] * len(ids2) + [0] * pad2], device=self.device)
        
        with torch.no_grad():
            e1 = self._get_embedding(ids1_tensor, mask1)
            e2 = self._get_embedding(ids2_tensor, mask2)
            return torch.sigmoid(self.model(e1, e2)).item()
    
    def predict_batch(self, target_seq: str, binder_seqs: List[str]) -> List[float]:
        """
        Predict interaction probabilities for multiple binders with the same target.
        
        More efficient than calling predict() multiple times because:
        1. Target embedding is precomputed via set_target() (called once at startup)
        2. Batched binder embedding computation
        3. Batched classifier inference
        
        Args:
            target_seq: Target protein sequence (should match the one set via set_target())
            binder_seqs: List of binder sequences
        
        Returns:
            List of interaction probabilities
        """
        if not binder_seqs:
            return []
        
        self.model.eval()
        max_aa = MAX_SEQ_LEN - 2
        batch_size = len(binder_seqs)
        
        # Use precomputed target embedding if available
        target_key = target_seq[:max_aa]
        
        if self._target_emb is not None and self._target_seq == target_key:
            # Use precomputed embedding
            target_emb = self._target_emb
        else:
            # Fallback: compute on-the-fly (shouldn't happen if set_target was called)
            target_ids = self.tokenizer.encode(target_key)
            target_pad = MAX_SEQ_LEN - len(target_ids)
            target_ids_tensor = torch.tensor([target_ids + [self.tokenizer.pad_id] * target_pad], device=self.device)
            target_mask = torch.tensor([[1] * len(target_ids) + [0] * target_pad], device=self.device)
            
            with torch.no_grad():
                target_emb = self._get_embedding(target_ids_tensor, target_mask)
        
        # Encode all binders
        binder_ids_list = []
        binder_masks_list = []
        for seq in binder_seqs:
            ids = self.tokenizer.encode(seq[:max_aa])
            pad = MAX_SEQ_LEN - len(ids)
            binder_ids_list.append(ids + [self.tokenizer.pad_id] * pad)
            binder_masks_list.append([1] * len(ids) + [0] * pad)
        
        binder_ids_tensor = torch.tensor(binder_ids_list, device=self.device)
        binder_masks_tensor = torch.tensor(binder_masks_list, device=self.device)
        
        binder_emb = None
        probs = None
        
        try:
            with torch.no_grad():
                # Expand target embedding to batch size
                target_emb_batch = target_emb.expand(batch_size, -1)
                
                # Get binder embeddings (batched)
                binder_emb = self._get_embedding(binder_ids_tensor, binder_masks_tensor)
                
                # Predict (batched)
                probs = torch.sigmoid(self.model(target_emb_batch, binder_emb))
                
                # Convert to numpy/list immediately to release GPU memory
                probs_squeezed = probs.squeeze(-1)
                if probs_squeezed.dim() == 0:
                    result = [probs_squeezed.item()]
                else:
                    result = probs_squeezed.cpu().numpy().tolist()
        finally:
            # Explicitly delete tensors to prevent memory leak
            del binder_ids_tensor, binder_masks_tensor
            if binder_emb is not None:
                del binder_emb
            if probs is not None:
                del probs
        
        return result


# ============================================================================
# Structure Filter Module (NetSurfP-3.0 based)
# ============================================================================ 
# Provides structure-aware filtering:
# - Trim disordered terminal regions
# - Filter by secondary structure content (helix/strand preference)
# - Detect compact structures (helix bundles, β-barrels)
#
# ============================================================================

# Check NetSurfP-3.0 availability
try:
    from nsp3.models import CNNbLSTM_ESM1b
    from nsp3.processing import PredictNSP3
    from nsp3.augmentation import string_token
    NSP3_AVAILABLE = True
except ImportError:
    NSP3_AVAILABLE = False
    print("Warning: NetSurfP-3.0 not available. Structure filtering will be skipped.")

# NetSurfP-3.0 model configuration
NSP3_MODEL_CONFIG = {
    'init_n_channels': 1280,
    'out_channels': 32,
    'cnn_layers': 2,
    'kernel_size': [129, 257],
    'padding': [64, 128],
    'n_hidden': 1024,
    'dropout': 0.5,
    'lstm_layers': 2,
    'embedding_args': {
        'arch': 'roberta_large',
        'dropout': 0.0,
        'attention_dropout': 0.0,
        'activation_dropout': 0.0,
        'ffn_embed_dim': 5120,
        'layers': 33,
        'attention_heads': 20,
        'embed_dim': 1280,
        'max_positions': 1024,
        'learned_pos': True,
        'activation_fn': 'gelu',
        'use_bert_init': True,
        'normalize_before': True,
        'preact_normalize': True,
        'normalize_after': True,
        'token_dropout': True,
        'no_seed_provided': False,
        'pooler_activation_fn': 'tanh',
        'pooler_dropout': 0.0,
        'checkpoint_transformer_block': False,
        'untie_weights_roberta': False
    },
    'embedding_pretrained': None
}

Q3_CLASS = {0: 'H', 1: 'E', 2: 'C'}
Q8_CLASS = {0: 'G', 1: 'H', 2: 'I', 3: 'B', 4: 'E', 5: 'S', 6: 'T', 7: 'C'}


@dataclass
class SSRegion:
    """A continuous secondary structure region"""
    start: int
    end: int
    ss_type: str  # 'H' for helix, 'E' for strand, 'C' for coil
    
    @property
    def length(self) -> int:
        return self.end - self.start
    
    def __contains__(self, idx: int) -> bool:
        return self.start <= idx < self.end


@dataclass
class StructureFilterResult:
    """Result of structure filtering operation."""
    # Sequence info
    original_sequence: str
    trimmed_sequence: str
    original_length: int
    trimmed_length: int
    
    # Trim info
    n_term_trim: int
    c_term_trim: int
    start_idx: int
    end_idx: int
    
    # Secondary structure
    ss3: str  # Trimmed SS3 string
    ss8: str  # Trimmed SS8 string
    
    # Metrics
    buried_fraction: float
    helix_fraction: float
    strand_fraction: float
    coil_fraction: float
    mean_disorder: float
    num_helix: int
    num_strand: int
    num_helix_strand: int
    ss_type: str  # 'helix' or 'strand' (major SS type)
    
    # Filter result
    passed: bool
    rejection_reasons: List[str]
    
    # Aliases for backward compatibility with result access
    @property
    def helix_content(self) -> float:
        return self.helix_fraction
    
    @property
    def strand_content(self) -> float:
        return self.strand_fraction
    
    @property
    def coil_content(self) -> float:
        return self.coil_fraction
    
    @property
    def structured_fraction(self) -> float:
        return self.helix_fraction + self.strand_fraction
    
    @property
    def ss_filter_passed(self) -> str:
        """Backward compatible: returns ss_type if passed, else 'rejected'"""
        return self.ss_type if self.passed else 'rejected'


class NetSurfP3Predictor:
    """NetSurfP-3.0 predictor for secondary structure and disorder."""
    
    def __init__(self, model_path: str, esm_path: str = None, device: str = None):
        if not NSP3_AVAILABLE:
            raise RuntimeError("NetSurfP-3.0 not available. Install with: pip install nsp3")
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = torch.device(device)
        self.model_path = model_path
        self.esm_path = esm_path
        
        if esm_path:
            NSP3_MODEL_CONFIG['embedding_pretrained'] = esm_path
        else:
            NSP3_MODEL_CONFIG['embedding_pretrained'] = os.path.expanduser(
                '~/.cache/torch/hub/checkpoints/esm1b_t33_650M_UR50S.pt'
            )
        
        self.model = CNNbLSTM_ESM1b(**NSP3_MODEL_CONFIG)
        model_data = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model_data['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.predictor = PredictNSP3(self.model, string_token, self.device)
        self._batch_count = 0
        print(f"Loaded NetSurfP-3.0 model on {self.device}: {model_path}")
    
    def reset_predictor(self):
        """Reset the internal predictor to clear any cached state."""
        del self.predictor
        gc.collect()
        self.predictor = PredictNSP3(self.model, string_token, self.device)
    
    def _clear_esm_cache(self):
        """Clear ESM1b internal representations cache to prevent memory leak.
        
        The ESM1b model stores 'representations' dict internally which accumulates
        memory across batches. This forces cleanup.
        """
        # Clear any cached representations in the ESM embedding layer
        if hasattr(self.model, 'embedding') and hasattr(self.model.embedding, 'model'):
            esm_model = self.model.embedding.model
            # Clear any internal state that might hold tensor references
            if hasattr(esm_model, 'layers'):
                for layer in esm_model.layers:
                    # Clear attention caches if present
                    if hasattr(layer, 'self_attn'):
                        if hasattr(layer.self_attn, '_cached_key'):
                            layer.self_attn._cached_key = None
                        if hasattr(layer.self_attn, '_cached_value'):
                            layer.self_attn._cached_value = None
    
    def _parse_prediction(self, seq: str, pred: np.ndarray) -> dict:
        """Parse raw prediction into structured result."""
        q8_prob = pred[:, 0:8]
        q3_prob = pred[:, 8:11]
        disorder = pred[:, 12]
        
        return {
            'sequence': seq,
            'ss3': ''.join([Q3_CLASS[i] for i in np.argmax(q3_prob, axis=1)]),
            'ss8': ''.join([Q8_CLASS[i] for i in np.argmax(q8_prob, axis=1)]),
            'rsa': pred[:, 13],
            'disorder': disorder,
            'phi': pred[:, 14],
            'psi': pred[:, 15],
            'q3_prob': q3_prob,
        }
    
    def predict(self, sequence: str) -> dict:
        """Predict structural features for a single sequence."""
        fasta = [(">protein", sequence)]
        identifiers, sequences, predictions = self.predictor(fasta)
        
        seq = sequences[0]
        
        # Force conversion to numpy to break any tensor references
        predictions_np = []
        for p in predictions:
            if hasattr(p, 'cpu'):  # It's a torch tensor
                predictions_np.append(p.cpu().numpy())
            elif hasattr(p, 'numpy'):
                predictions_np.append(np.array(p))
            else:
                predictions_np.append(np.asarray(p))
        
        del predictions
        
        pred = np.concatenate([p[0][:len(seq)] for p in predictions_np], axis=1)
        del predictions_np
        
        return self._parse_prediction(seq, pred)
    
    def predict_batch(self, sequences: List[str]) -> List[dict]:
        """
        Predict structural features for multiple sequences in batch.
        
        More efficient than calling predict() multiple times because:
        1. Single pass through ESM embedding model
        2. Batched CNN-LSTM inference
        
        Args:
            sequences: List of protein sequences
        
        Returns:
            List of prediction dictionaries
        """
        if not sequences:
            return []
        
        # Periodically reset predictor to clear any internal caches (every 100 batches)
        self._batch_count += 1
        if self._batch_count % 100 == 0:
            self.reset_predictor()
            torch_gc()
        
        # Create FASTA-like input
        fasta = [(f">seq_{i}", seq) for i, seq in enumerate(sequences)]
        identifiers, result_seqs, predictions = self.predictor(fasta)
        
        # predictions is a list of arrays, one per output type (q8, q3, disorder, etc.)
        # Each array has shape (batch_size, max_seq_len, features)
        # IMPORTANT: Force conversion to numpy to break any tensor references
        predictions_np = []
        for p in predictions:
            if hasattr(p, 'cpu'):  # It's a torch tensor
                predictions_np.append(p.cpu().numpy())
            elif hasattr(p, 'numpy'):  # It's a tensor-like object
                predictions_np.append(np.array(p))
            else:
                predictions_np.append(np.asarray(p))
        
        # Delete original predictions immediately
        del predictions
        
        # Clear ESM1b internal cache to prevent memory accumulation
        self._clear_esm_cache()
        
        results = []
        for i, seq in enumerate(result_seqs):
            seq_len = len(seq)
            # Extract predictions for this sequence from each output array
            pred = np.concatenate([p[i][:seq_len] for p in predictions_np], axis=1)
            results.append(self._parse_prediction(seq, pred))
        
        # Clean up
        del predictions_np
        
        return results


def find_ss_regions(ss3: str) -> List[SSRegion]:
    """Find all continuous secondary structure regions."""
    if not ss3:
        return []
    
    regions = []
    current_type = ss3[0]
    start = 0
    
    for i, c in enumerate(ss3):
        if c != current_type:
            regions.append(SSRegion(start, i, current_type))
            current_type = c
            start = i
    
    regions.append(SSRegion(start, len(ss3), current_type))
    return regions


def trim_terminals(
    sequence: str,
    ss3: str,
    ss8: str,
    disorder: np.ndarray,
    rsa: np.ndarray,
    merge_gap: int = 3,
    min_ss_length: int = 5,
    max_flank_coil: int = 10,
    n_buffer: int = 2,
    c_buffer: int = 3,
    big_coil_threshold: int = 15,
) -> Tuple[str, str, str, np.ndarray, np.ndarray, int, int]:
    """
    Trim N and C terminal regions using structure-aware procedure.
    
    Procedure:
    1. Trim N and C terminal coil
    2. Merge helix/strand segments with gap coil <= merge_gap
    3. For N terminal: forward loop through merged segments, fix when segment > min_ss_length 
       and right flanking coil <= max_flank_coil
    4. For C terminal: backward loop through merged segments, fix when segment > min_ss_length
       and left flanking coil <= max_flank_coil
    5. Apply n_buffer (2 AA) at N-term and c_buffer (3 AA) at C-term
    6. Find biggest coil segment; if > big_coil_threshold, keep only the larger fragment
       (left or right of the big coil)
    
    Args:
        sequence: Protein sequence
        ss3: 3-state secondary structure string (H, E, C)
        ss8: 8-state secondary structure string
        disorder: Disorder probability array
        rsa: Relative solvent accessibility array
        merge_gap: Maximum coil gap to merge adjacent SS segments (default: 3)
        min_ss_length: Minimum SS segment length to anchor trim site (default: 5)
        max_flank_coil: Maximum flanking coil length allowed (default: 10)
        n_buffer: Buffer residues to keep at N-terminus (default: 2)
        c_buffer: Buffer residues to keep at C-terminus (default: 3)
        big_coil_threshold: If biggest coil > this, split and keep larger fragment (default: 15)
    
    Returns:
        trimmed_sequence, trimmed_ss3, trimmed_ss8, trimmed_disorder, trimmed_rsa,
        start_idx, end_idx
    """
    n = len(sequence)
    if n == 0:
        return sequence, ss3, ss8, disorder, rsa, 0, 0
    
    # Step 1: Find all SS regions
    regions = find_ss_regions(ss3)
    
    # Separate SS (H/E) and coil (C) regions
    ss_regions = [r for r in regions if r.ss_type in ('H', 'E')]
    
    if len(ss_regions) == 0:
        # No SS elements, return middle portion
        start_idx = n // 4
        end_idx = 3 * n // 4
        return (
            sequence[start_idx:end_idx],
            ss3[start_idx:end_idx],
            ss8[start_idx:end_idx],
            disorder[start_idx:end_idx],
            rsa[start_idx:end_idx],
            start_idx,
            end_idx
        )
    
    # Step 2: Merge SS segments with gap coil <= merge_gap
    # Create merged segments: each is (start, end, total_ss_length)
    merged_segments = []
    current_start = ss_regions[0].start
    current_end = ss_regions[0].end
    
    for i in range(1, len(ss_regions)):
        prev_ss = ss_regions[i - 1]
        curr_ss = ss_regions[i]
        gap = curr_ss.start - prev_ss.end  # Coil gap between SS regions
        
        if gap <= merge_gap:
            # Merge: extend current segment
            current_end = curr_ss.end
        else:
            # Don't merge: save current segment and start new one
            merged_segments.append((current_start, current_end, current_end - current_start))
            current_start = curr_ss.start
            current_end = curr_ss.end
    
    # Don't forget the last segment
    merged_segments.append((current_start, current_end, current_end - current_start))
    
    # Step 3: Find N-terminal trim site (forward loop)
    # Look for first merged segment with length > min_ss_length and right flanking coil <= max_flank_coil
    n_trim_idx = 0  # Default: start of sequence
    
    for i, (seg_start, seg_end, seg_len) in enumerate(merged_segments):
        if seg_len > min_ss_length:
            # Check right flanking coil length
            if i < len(merged_segments) - 1:
                next_seg_start = merged_segments[i + 1][0]
                right_flank_coil = next_seg_start - seg_end
            else:
                # Last segment, no right flank to worry about
                right_flank_coil = 0
            
            if right_flank_coil <= max_flank_coil:
                # Fix N-terminal at this segment's start
                n_trim_idx = seg_start
                break
        # Else: move to next segment
    
    # Step 4: Find C-terminal trim site (backward loop)
    # Look for first merged segment (from right) with length > min_ss_length and left flanking coil <= max_flank_coil
    c_trim_idx = n  # Default: end of sequence
    
    for i in range(len(merged_segments) - 1, -1, -1):
        seg_start, seg_end, seg_len = merged_segments[i]
        if seg_len > min_ss_length:
            # Check left flanking coil length
            if i > 0:
                prev_seg_end = merged_segments[i - 1][1]
                left_flank_coil = seg_start - prev_seg_end
            else:
                # First segment, no left flank to worry about
                left_flank_coil = 0
            
            if left_flank_coil <= max_flank_coil:
                # Fix C-terminal at this segment's end
                c_trim_idx = seg_end
                break
        # Else: move to previous segment
    
    # Step 5: Apply buffers
    # N-terminal: keep n_buffer residues before the trim site
    start_idx = max(0, n_trim_idx - n_buffer)
    # C-terminal: keep c_buffer residues after the trim site
    end_idx = min(n, c_trim_idx + c_buffer)
    
    # Ensure valid range
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = n
    
    # Step 6: Find biggest coil segment in the trimmed region
    # If big_coil_seg > big_coil_threshold, keep only the larger fragment
    trimmed_ss3 = ss3[start_idx:end_idx]
    trimmed_regions = find_ss_regions(trimmed_ss3)
    coil_regions = [r for r in trimmed_regions if r.ss_type == 'C']
    
    if coil_regions:
        # Find the biggest coil segment
        biggest_coil = max(coil_regions, key=lambda r: r.length)
        
        if biggest_coil.length > big_coil_threshold:
            # Split at the biggest coil and keep the larger fragment
            # Left fragment: from start to beginning of big coil
            left_len = biggest_coil.start
            # Right fragment: from end of big coil to end
            right_len = len(trimmed_ss3) - biggest_coil.end
            
            if left_len >= right_len:
                # Keep left fragment
                end_idx = start_idx + biggest_coil.start
            else:
                # Keep right fragment
                start_idx = start_idx + biggest_coil.end
    
    # Final validation
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = n
    
    return (
        sequence[start_idx:end_idx],
        ss3[start_idx:end_idx],
        ss8[start_idx:end_idx],
        disorder[start_idx:end_idx],
        rsa[start_idx:end_idx],
        start_idx,
        end_idx
    )


def calculate_metrics(
    ss3: str,
    disorder: np.ndarray,
    rsa: np.ndarray,
    burial_threshold: float = 0.25
) -> dict:

    n = len(ss3)
    if n == 0:
        return {
            'buried_fraction': 0.0,
            'helix_fraction': 0.0,
            'strand_fraction': 0.0,
            'coil_fraction': 1.0,
            'mean_disorder': 1.0,
            'num_helix': 0,
            'num_strand': 0,
            'num_helix_strand': 0,
            'ss_type': 'coil',
        }
    
    # SS fractions
    helix_count = ss3.count('H')
    strand_count = ss3.count('E')
    coil_count = ss3.count('C')
    
    helix_fraction = helix_count / n
    strand_fraction = strand_count / n
    coil_fraction = coil_count / n
    
    # Determine SS type based on dominance within structured region
    # helix if helix_fraction/structured_fraction > 0.75
    # strand if strand_fraction/structured_fraction > 0.75
    # mixed otherwise
    structured_fraction = helix_fraction + strand_fraction
    if structured_fraction > 0:
        helix_ratio = helix_fraction / structured_fraction
        strand_ratio = strand_fraction / structured_fraction
        if helix_ratio > 0.75:
            ss_type = 'helix'
        elif strand_ratio > 0.75:
            ss_type = 'strand'
        else:
            ss_type = 'mixed'
    else:
        ss_type = 'coil'
    
    # Count SS elements
    regions = find_ss_regions(ss3)
    num_helix = sum(1 for r in regions if r.ss_type == 'H')
    num_strand = sum(1 for r in regions if r.ss_type == 'E')
    num_helix_strand = num_helix + num_strand
    
    # Disorder
    mean_disorder = float(np.mean(disorder))
    
    # Burial
    buried_fraction = float(np.mean(rsa < burial_threshold))
    
    return {
        'buried_fraction': buried_fraction,
        'helix_fraction': helix_fraction,
        'strand_fraction': strand_fraction,
        'coil_fraction': coil_fraction,
        'mean_disorder': mean_disorder,
        'num_helix': num_helix,
        'num_strand': num_strand,
        'num_helix_strand': num_helix_strand,
        'ss_type': ss_type,
    }


def apply_filter_criteria(
    seq_length: int,
    metrics: dict,
    min_length: int = 70,
    max_length: int = 130,
    min_buried_fraction: float = 0.25,
    min_structured_fraction: float = 0.6,
    max_mean_disorder: float = 0.1,
    min_num_helix_strand: int = 2,
    max_num_helix_strand: int = 8,
    ss_preference: str = None,
) -> Tuple[bool, List[str]]:
    """
    Apply filter criteria.
    
    Criteria:
        - 70 < seq_length < 130
        - buried_fraction > 0.25
        - structured_fraction (H + E) >= 0.6
        - mean_disorder < 0.1
        - 2 <= num_helix_strand <= 8
        - ss_preference: None (accept all), 'helix' (only helix), 'strand' (only strand)
    
    Returns:
        (passed, rejection_reasons)
    """
    rejection_reasons = []
    
    if seq_length <= min_length:
        rejection_reasons.append(f"seq_length={seq_length} <= {min_length}")
    if seq_length >= max_length:
        rejection_reasons.append(f"seq_length={seq_length} >= {max_length}")
    
    if metrics['buried_fraction'] <= min_buried_fraction:
        rejection_reasons.append(
            f"buried_fraction={metrics['buried_fraction']:.3f} <= {min_buried_fraction}"
        )
    
    # Check structured_fraction (helix + strand)
    structured_fraction = metrics['helix_fraction'] + metrics['strand_fraction']
    if structured_fraction < min_structured_fraction:
        rejection_reasons.append(
            f"structured_fraction={structured_fraction:.3f} < {min_structured_fraction}"
        )
    
    if metrics['mean_disorder'] >= max_mean_disorder:
        rejection_reasons.append(
            f"mean_disorder={metrics['mean_disorder']:.3f} >= {max_mean_disorder}"
        )
    
    if metrics['num_helix_strand'] < min_num_helix_strand:
        rejection_reasons.append(
            f"num_helix_strand={metrics['num_helix_strand']} < {min_num_helix_strand}"
        )
    if metrics['num_helix_strand'] > max_num_helix_strand:
        rejection_reasons.append(
            f"num_helix_strand={metrics['num_helix_strand']} > {max_num_helix_strand}"
        )
    
    # SS preference filter
    # ss_preference=None: accept all (helix, strand, mixed)
    # ss_preference='helix': only accept ss_type='helix'
    # ss_preference='strand': only accept ss_type='strand'
    if ss_preference is not None:
        ss_type = metrics.get('ss_type', 'coil')
        if ss_preference == 'helix' and ss_type != 'helix':
            rejection_reasons.append(
                f"ss_preference=helix but ss_type={ss_type}"
            )
        elif ss_preference == 'strand' and ss_type != 'strand':
            rejection_reasons.append(
                f"ss_preference=strand but ss_type={ss_type}"
            )
    
    passed = len(rejection_reasons) == 0
    return passed, rejection_reasons


class StructureFilter:
    """
    Integrated structure-aware filter using NetSurfP-3.0.
    
    Workflow:
        1. Run NetSurfP-3.0 predictions
        2. Trim terminals:
           - Trim N/C terminal coils
           - Merge SS segments with gap coil <= 3
           - Find N-term: forward loop, fix when segment > 5 and right flank coil <= 10
           - Find C-term: backward loop, fix when segment > 5 and left flank coil <= 10
           - Apply buffers (2 AA at N-term, 3 AA at C-term)
           - If biggest coil > 15, keep only larger fragment
        3. Calculate metrics
        4. Apply filter criteria
    """
    
    def __init__(
        self,
        nsp3_model_path: str,
        esm_path: str = None,
        device: str = None,  # Auto-detect if None
        # Trimming parameters
        merge_gap: int = 3,
        min_ss_length: int = 5,
        max_flank_coil: int = 10,
        n_buffer: int = 2,
        c_buffer: int = 3,
        big_coil_threshold: int = 15,
        # Filter criteria
        min_length: int = 70,
        max_length: int = 130,
        min_buried_fraction: float = 0.25,
        min_structured_fraction: float = 0.6,
        max_mean_disorder: float = 0.1,
        min_num_helix_strand: int = 2,
        max_num_helix_strand: int = 8,
        burial_threshold: float = 0.25,
        ss_preference: str = None,  # 'helix', 'strand', or None
    ):
        self.predictor = NetSurfP3Predictor(nsp3_model_path, esm_path, device)
        
        # Trimming params
        self.merge_gap = merge_gap
        self.min_ss_length = min_ss_length
        self.max_flank_coil = max_flank_coil
        self.n_buffer = n_buffer
        self.c_buffer = c_buffer
        self.big_coil_threshold = big_coil_threshold
        
        # Filter criteria
        self.min_length = min_length
        self.max_length = max_length
        self.min_buried_fraction = min_buried_fraction
        self.min_structured_fraction = min_structured_fraction
        self.max_mean_disorder = max_mean_disorder
        self.min_num_helix_strand = min_num_helix_strand
        self.max_num_helix_strand = max_num_helix_strand
        self.burial_threshold = burial_threshold
        self.ss_preference = ss_preference
    
    def _process_single(self, nsp3_result: dict, ss_preference: str = None) -> StructureFilterResult:
        """Process a single NetSurfP-3.0 prediction."""
        seq = nsp3_result['sequence']
        ss3 = nsp3_result['ss3']
        ss8 = nsp3_result['ss8']
        disorder = nsp3_result['disorder']
        rsa = nsp3_result['rsa']
        
        # Step 2: Trim terminals
        trimmed_seq, trimmed_ss3, trimmed_ss8, trimmed_disorder, trimmed_rsa, start_idx, end_idx = trim_terminals(
            seq, ss3, ss8, disorder, rsa,
            merge_gap=self.merge_gap,
            min_ss_length=self.min_ss_length,
            max_flank_coil=self.max_flank_coil,
            n_buffer=self.n_buffer,
            c_buffer=self.c_buffer,
            big_coil_threshold=self.big_coil_threshold,
        )
        
        # Step 3: Calculate metrics
        metrics = calculate_metrics(
            trimmed_ss3, trimmed_disorder, trimmed_rsa,
            burial_threshold=self.burial_threshold
        )
        
        # Step 4: Apply filter criteria
        passed, rejection_reasons = apply_filter_criteria(
            seq_length=len(trimmed_seq),
            metrics=metrics,
            min_length=self.min_length,
            max_length=self.max_length,
            min_buried_fraction=self.min_buried_fraction,
            min_structured_fraction=self.min_structured_fraction,
            max_mean_disorder=self.max_mean_disorder,
            min_num_helix_strand=self.min_num_helix_strand,
            max_num_helix_strand=self.max_num_helix_strand,
            ss_preference=ss_preference,
        )
        
        return StructureFilterResult(
            original_sequence=seq,
            trimmed_sequence=trimmed_seq,
            original_length=len(seq),
            trimmed_length=len(trimmed_seq),
            n_term_trim=start_idx,
            c_term_trim=len(seq) - end_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            ss3=trimmed_ss3,
            ss8=trimmed_ss8,
            buried_fraction=metrics['buried_fraction'],
            helix_fraction=metrics['helix_fraction'],
            strand_fraction=metrics['strand_fraction'],
            coil_fraction=metrics['coil_fraction'],
            mean_disorder=metrics['mean_disorder'],
            num_helix=metrics['num_helix'],
            num_strand=metrics['num_strand'],
            num_helix_strand=metrics['num_helix_strand'],
            ss_type=metrics['ss_type'],
            passed=passed,
            rejection_reasons=rejection_reasons,
        )
    
    def filter_batch(self, sequences: List[str], ss_preference: str = None) -> List[StructureFilterResult]:

        if not sequences:
            return []
        
        # Use instance default if not specified
        if ss_preference is None:
            ss_preference = self.ss_preference
        
        # Step 1: Batch predict
        nsp3_results = self.predictor.predict_batch(sequences)
        
        # Steps 2-4: Process each prediction
        results = []
        for nsp3_result in nsp3_results:
            result = self._process_single(nsp3_result, ss_preference)
            results.append(result)
        
        return results
    
    def filter_single(self, sequence: str, ss_preference: str = None) -> StructureFilterResult:
        """Filter a single sequence."""
        results = self.filter_batch([sequence], ss_preference)
        return results[0] if results else None
    
    def get_passing_sequences(
        self, 
        sequences: List[str],
        ss_preference: str = None
    ) -> Tuple[List[str], List[StructureFilterResult]]:
        """
        Filter sequences and return only those that pass.
        
        Args:
            sequences: List of protein sequences
            ss_preference: 'helix', 'strand', or None. If None, uses instance default.
        
        Returns:
            (passing_sequences, all_results)
        """
        results = self.filter_batch(sequences, ss_preference)
        passing = [r.trimmed_sequence for r in results if r.passed]
        return passing, results


# Convenience function for one-off filtering
def filter_sequences(
    sequences: List[str],
    nsp3_model_path: str,
    esm_path: str = None,
    device: str = 'cpu',
    **filter_kwargs
) -> List[StructureFilterResult]:
    """
    Convenience function to filter sequences without manually creating a StructureFilter.
    
    Args:
        sequences: List of protein sequences
        nsp3_model_path: Path to NetSurfP-3.0 model weights
        esm_path: Optional path to ESM-1b model
        device: 'cpu' or 'cuda'
        **filter_kwargs: Additional arguments for StructureFilter
    
    Returns:
        List of StructureFilterResult objects
    """
    structure_filter = StructureFilter(
        nsp3_model_path=nsp3_model_path,
        esm_path=esm_path,
        device=device,
        **filter_kwargs
    )
    return structure_filter.filter_batch(sequences)



# ============================================================================
# Parallel Generation Worker (module-level for ProcessPoolExecutor)
# ============================================================================

def _batch_generation_worker(args_tuple):
    """
    Worker function for parallel batch binder generation.
    
    Each worker:
    1. Loads its own copy of all models on assigned GPU
    2. Runs batch generation + batch filtering pipeline
    3. Adds results to shared queue until global target is reached
    
    No structure prediction here - that runs after all workers finish.
    
    Args:
        args_tuple: (config_dict, shared_counter, shared_lock, result_queue) or (config_dict,)
            - config_dict: all parameters needed
            - shared_counter: multiprocessing.Value for tracking total designs (optional)
            - shared_lock: multiprocessing.Lock for thread-safe counter updates (optional)
            - result_queue: multiprocessing.Queue for results (optional)
    
    Returns:
        List of result dicts for successful designs (only used if no shared_counter)
    """
    # Handle both old (static) and new (dynamic) calling conventions
    if len(args_tuple) == 4:
        config, shared_counter, shared_lock, result_queue = args_tuple
        use_dynamic = True
    else:
        config = args_tuple[0]
        shared_counter = None
        shared_lock = None
        result_queue = None
        use_dynamic = False
    
    gpu_id = config.get('gpu_id', 0)
    accelerator_type = config.get('accelerator_type', 'cpu')
    worker_id = config.get('worker_id', 0)
    num_to_generate = config.get('num_to_generate', 10)  # Global target for dynamic mode
    
    # Setup GPU IMMEDIATELY before any CUDA operations
    # This must happen before torch initializes CUDA
    if accelerator_type == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        # Force PyTorch to see only this GPU
        import torch
        torch.cuda.set_device(0)  # After CUDA_VISIBLE_DEVICES, GPU 0 = gpu_id
        device = 'cuda'  # Use 'cuda' not 'cuda:0' for consistency
        print(f"[Worker {worker_id}] Using GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")
    elif accelerator_type == 'mps':
        device = 'mps'
        print(f"[Worker {worker_id}] Using MPS")
    else:
        device = 'cpu'
        print(f"[Worker {worker_id}] Using CPU")
    
    try:
        # ===================================================================
        # Load models (each worker has its own copy)
        # ===================================================================
        print(f"[Worker {worker_id}] Loading models...")
        
        # Load LLM with generation parameters from config
        chat_model = None
        if config.get('gen_model_path'):
            chat_model = HuggingfaceEngine(
                model_path=config['gen_model_path'],
                temperature=config.get('temperature', 0.6),
                top_k=config.get('top_k', 40),
                top_p=config.get('top_p', 0.9),
                repetition_penalty=1.2,
                max_new_tokens=config.get('max_length', 150) // 2,
                min_new_tokens=config.get('min_length', 50) // 2,
            )
            print(f"[Worker {worker_id}] Loaded LLM (temp={config.get('temperature', 0.6)}, top_k={config.get('top_k', 40)}, top_p={config.get('top_p', 0.9)})")
        
        # Load Structure Filter (NetSurfP-3.0)
        structure_filter = None
        if config.get('nsp3_model_path') and config.get('use_structure_filter'):
            structure_filter = StructureFilter(
                nsp3_model_path=config['nsp3_model_path'],
                esm_path=config.get('nsp3_esm_path'),
                device=device,
                min_structured_fraction=config.get('min_structured_fraction', 0.6),
                max_mean_disorder=config.get('max_mean_disorder', 0.10),
                min_buried_fraction=config.get('min_buried_fraction', 0.25),
            )
            print(f"[Worker {worker_id}] Loaded NetSurfP-3.0")
        
        # Load PPI Predictor (uses ESM2-8M)
        ppi_predictor = None
        if config.get('ppi_model_path'):
            ppi_predictor = SiamesePPIPredictor(
                device=torch.device(device)
            )
            ppi_predictor.load(config['ppi_model_path'])
            # Precompute target embedding once at startup
            ppi_predictor.set_target(config['target_seq'])
            print(f"[Worker {worker_id}] Loaded PPI predictor (ESM2-8M)")
        
        # Report GPU memory usage
        if accelerator_type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[Worker {worker_id}] GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # Setup QC config
        qc_config = QCConfig(
            min_length=config.get('min_length', 50),
            max_length=config.get('max_length', 150)
        )
        
        target_seq = config['target_seq']
        ppi_threshold = config.get('ppi_threshold', 0.3)
        ss_preference = config.get('ss_preference')
        generate_batch_size = config.get('generate_batch_size', 8)
        
        # Truncate target sequence for LLM prompt
        max_len = HuggingfaceEngine.MAX_TARGET_SEQ_LENGTH
        truncated_seq = target_seq[:max_len] if len(target_seq) > max_len else target_seq
        
        # ===================================================================
        # Generation loop - batch generate until we have enough
        # ===================================================================
        local_results = []  # Results from this worker
        total_generated = 0
        batch_count = 0
        max_batches = num_to_generate * 10000  # Safety limit (increased for strict filters)
        
        # Helper function to check if we should stop
        def should_stop():
            if use_dynamic and shared_counter is not None:
                with shared_lock:
                    return shared_counter.value >= num_to_generate
            else:
                return len(local_results) >= num_to_generate
        
        # Helper function to add results
        def add_results(new_results):
            nonlocal local_results
            if use_dynamic and shared_counter is not None:
                added = []
                with shared_lock:
                    for r in new_results:
                        if shared_counter.value >= num_to_generate:
                            break
                        shared_counter.value += 1
                        r['id'] = shared_counter.value - 1  # Assign global ID
                        added.append(r)
                        result_queue.put(r)
                local_results.extend(added)
                return len(added)
            else:
                local_results.extend(new_results)
                return len(new_results)
        
        print(f"[Worker {worker_id}] Starting generation (global target: {num_to_generate} designs)...")
        
        while not should_stop() and batch_count < max_batches:
            batch_count += 1
            
            # ---------------------------------------------------------
            # Step 1: Batch LLM generation
            # ---------------------------------------------------------
            query = f'[Generate Binder] Seq=<{truncated_seq}>'
            messages = [{"role": "user", "content": query}]
            
            raw_responses = chat_model.batch_chat(messages, generate_batch_size)
            
            # Clean up sequences
            raw_sequences = []
            for response in raw_responses:
                seq = response.replace('Seq=<', '').replace('>', '').strip()
                clean_seq = ''.join(c for c in seq.upper() if c in VALID_AAS)
                if clean_seq:
                    raw_sequences.append(clean_seq)
            
            total_generated += len(raw_sequences)
            
            if not raw_sequences:
                continue
            
            # ---------------------------------------------------------
            # Step 2: QC1 - Fast pre-filter (alphabet, length)
            # ---------------------------------------------------------
            qc1_passed = []
            for seq in raw_sequences:
                qc_result = qc1_prefilter(seq, qc_config)
                if qc_result.passed_all:
                    qc1_passed.append(seq)
            
            if not qc1_passed:
                continue
            
            # ---------------------------------------------------------
            # Step 3: Structure Filter (NetSurfP-3.0) - Batch
            # ---------------------------------------------------------
            if structure_filter is not None:
                sf_results = structure_filter.filter_batch(qc1_passed, ss_preference)
                
                sf_passed_seqs = []
                sf_passed_results = []
                sf_passed_original = []
                
                for seq, sf_result in zip(qc1_passed, sf_results):
                    if sf_result.passed:
                        sf_passed_seqs.append(sf_result.trimmed_sequence)
                        sf_passed_results.append(sf_result)
                        sf_passed_original.append(seq)
                
                # Clean up sf_results to release any held references
                del sf_results
                
                # Force GPU cleanup after structure prediction
                torch_gc()
                
                if not sf_passed_seqs:
                    continue
            else:
                sf_passed_seqs = qc1_passed
                sf_passed_results = [None] * len(qc1_passed)
                sf_passed_original = qc1_passed
            
            # ---------------------------------------------------------
            # Step 4: QC2 - Post-trim quality checks
            # ---------------------------------------------------------
            qc2_passed_seqs = []
            qc2_passed_results = []
            qc2_sf_results = []
            qc2_original = []
            
            for seq, sf_result, orig_seq in zip(sf_passed_seqs, sf_passed_results, sf_passed_original):
                qc_result = qc2_postfilter(seq, qc_config)
                if qc_result.passed_all:
                    qc2_passed_seqs.append(seq)
                    qc2_passed_results.append(qc_result)
                    qc2_sf_results.append(sf_result)
                    qc2_original.append(orig_seq)
            
            if not qc2_passed_seqs:
                continue
            
            # ---------------------------------------------------------
            # Step 5: PPI Prediction - Batch
            # ---------------------------------------------------------
            if ppi_predictor is not None:
                # Predict PPI for trimmed sequences
                ppi_probs = ppi_predictor.predict_batch(target_seq, qc2_passed_seqs)
                
                batch_results = []
                for seq, ppi_prob, qc_result, sf_result, orig_seq in zip(
                    qc2_passed_seqs, ppi_probs, qc2_passed_results, qc2_sf_results, qc2_original
                ):
                    if ppi_prob >= ppi_threshold:
                        # Build result dict
                        result = {
                            'worker_id': worker_id,
                            'seq_original': orig_seq,
                            'seq_trimmed': seq,
                            'ppi_prob': ppi_prob,
                            'length': len(seq),
                            'max_aa_percent': qc_result.max_single_aa_percent,
                            'hydrophobic_pct': qc_result.hydrophobic_percent,
                            'aromatic_pct': qc_result.aromatic_percent,
                            'charged_pct': qc_result.charged_percent,
                            'entropy': qc_result.shannon_entropy,
                            'net_charge': qc_result.net_charge,
                        }
                        
                        if sf_result is not None:
                            result.update({
                                'helix_content': sf_result.helix_content,
                                'strand_content': sf_result.strand_content,
                                'coil_content': sf_result.coil_content,
                                'ss_type': sf_result.ss_filter_passed,
                                'ss3': sf_result.ss3,
                                'n_term_trim': sf_result.n_term_trim,
                                'c_term_trim': sf_result.c_term_trim,
                                # Compact core metrics
                                'mean_disorder': sf_result.mean_disorder,
                                'buried_fraction': sf_result.buried_fraction,
                                'structured_fraction': sf_result.structured_fraction,
                            })
                        
                        batch_results.append(result)
                
                # Explicitly clean up prediction results to prevent memory accumulation
                del ppi_probs
                
                # Force GPU cleanup after PPI prediction
                torch_gc()
                
                # Add results using helper (handles shared counter if in dynamic mode)
                if batch_results:
                    add_results(batch_results)
                
                # Check if we should stop
                if should_stop():
                    break
            
            # Progress update every 100 batches (reduced frequency)
            if batch_count % 100 == 0:
                if use_dynamic and shared_counter is not None:
                    with shared_lock:
                        global_count = shared_counter.value
                    print(f"[Worker {worker_id}] Batch {batch_count}: contributed {len(local_results)}, global {global_count}/{num_to_generate}")
                else:
                    print(f"[Worker {worker_id}] Batch {batch_count}: {len(local_results)}/{num_to_generate} designs")
        
        # Check if we hit the batch limit
        if batch_count >= max_batches and not should_stop():
            pass_rate = len(local_results) / max(total_generated, 1) * 100
            print(f"[Worker {worker_id}] ⚠️  Hit max batch limit ({max_batches}). Pass rate: {pass_rate:.2f}%")
            print(f"[Worker {worker_id}] Consider: relaxing filters or increasing --num_designs expectation")
        
        print(f"[Worker {worker_id}] Finished: contributed {len(local_results)} designs from {total_generated} generated")
        
        # Cleanup
        del chat_model, structure_filter, ppi_predictor
        torch_gc()
        
        return local_results
        
    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return []


class BinderDesignPipeline:
    """
    Binder design pipeline with structure-aware filtering.
    
    Workflow (per worker):
    1. Generate binder sequences using LLM (batch generation)
    2. QC1 (fast checks: length, composition) - batch
    3. Structure Filter (NetSurfP-3.0: trim disorder, SS content, pLDDT) - batch
    4. QC2 (post-trim checks) - batch
    5. PPI Classifier - batch
    
    Output: Filtered binder sequences.
    Supports parallel generation across multiple GPUs.
    """

    def __init__(
        self,
        target_seq: str,
        gen_model_path: str = None,
        ppi_model_path: str = None,
        output_dir: str = 'output',
        ppi_threshold: float = 0.3,
        min_length: int = 50,
        max_length: int = 150,
        device: str = None,
        # LLM generation parameters
        temperature: float = 0.6,
        top_k: int = 40,
        top_p: float = 0.9,
        # Structure Filter parameters
        nsp3_model_path: str = None,
        nsp3_esm_path: str = None,
        use_structure_filter: bool = True,
        ss_preference: str = None,  # 'helix', 'strand', or None (either)
        min_structured_fraction: float = 0.6,
        max_mean_disorder: float = 0.10,
        min_buried_fraction: float = 0.25,
        # Batch generation
        generate_batch_size: int = 8,  # sequences per LLM call
        # Parallel processing (multi-GPU)
        parallel_workers: int = 1,
        gpu_list: List[int] = None,
        accelerator_type: str = 'cpu',  # 'cuda', 'mps', or 'cpu'
    ):
        self.target_seq = target_seq
        self.output_dir = output_dir
        self.ppi_threshold = ppi_threshold
        self.min_length = min_length
        self.max_length = max_length
        
        # LLM generation parameters
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # Batch generation settings
        self.generate_batch_size = generate_batch_size
        
        # Parallel processing settings
        self.parallel_workers = max(1, parallel_workers)
        self.gpu_list = gpu_list if gpu_list else [0]
        self.accelerator_type = accelerator_type
        
        # Store model paths
        self.gen_model_path = gen_model_path
        self.ppi_model_path = ppi_model_path
        self.nsp3_model_path = nsp3_model_path
        self.nsp3_esm_path = nsp3_esm_path
        
        # Structure filter settings
        self.ss_preference = ss_preference
        self.min_structured_fraction = min_structured_fraction
        self.use_structure_filter = use_structure_filter and NSP3_AVAILABLE
        self.max_mean_disorder = max_mean_disorder
        self.min_buried_fraction = min_buried_fraction
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Set device
        self.device = torch.device(device) if device else get_device()
        
        # Initialize models (only for single-worker mode)
        self.chat_model = None
        self.ppi_predictor = None
        self.structure_filter = None
        self.qc_config = QCConfig(min_length=min_length, max_length=max_length)
        
        # Only load models in main process for single-worker mode
        # For parallel mode, each worker loads its own models
        if self.parallel_workers == 1:
            # Load generation model with max_new_tokens based on max_length
            if gen_model_path and TRANSFORMERS_AVAILABLE:
                print(f"\nLoading sequence generation model from: {gen_model_path}")
                self.chat_model = HuggingfaceEngine(
                    model_path=gen_model_path,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=1.2,
                    max_new_tokens=max_length // 2,
                    min_new_tokens=min_length // 2,
                )
            
            # Load PPI predictor (uses ESM2-8M)
            if ppi_model_path:
                print(f"\nLoading ESM2-8M for PPI prediction...")
                self.ppi_predictor = SiamesePPIPredictor(device=self.device)
                print(f"Loading PPI model from: {ppi_model_path}")
                self.ppi_predictor.load(ppi_model_path)
                # Precompute target embedding once at startup
                self.ppi_predictor.set_target(target_seq)
            
            # Load Structure Filter (NetSurfP-3.0)
            if self.use_structure_filter and nsp3_model_path and NSP3_AVAILABLE:
                print(f"\nLoading Structure Filter (NetSurfP-3.0) from: {nsp3_model_path}")
                # Use same device as other models
                device_str = str(self.device)
                self.structure_filter = StructureFilter(
                    nsp3_model_path=nsp3_model_path,
                    esm_path=nsp3_esm_path,
                    device=device_str,
                    min_structured_fraction=min_structured_fraction,
                    max_mean_disorder=max_mean_disorder,
                    min_buried_fraction=min_buried_fraction,
                )
                pref_str = ss_preference if ss_preference else 'all (helix/strand/mixed)'
                print(f"Structure Filter initialized on {device_str}: SS preference={pref_str}, min_structured_fraction={min_structured_fraction*100:.0f}%")
        else:
            print(f"\nParallel mode: {self.parallel_workers} workers on GPUs {self.gpu_list}")
            print("Models will be loaded in worker processes.")
        
        # Show warning for structure filter if not available
        if self.use_structure_filter and not NSP3_AVAILABLE:
            print("\nWarning: Structure Filter requested but NetSurfP-3.0 not available.")
            print("         Install with: pip install nsp3")
            print("         Continuing without structure filtering...")
            self.use_structure_filter = False
        elif self.use_structure_filter and not nsp3_model_path:
            # No nsp3_model provided - show strong warning
            print("\n" + "!"*80)
            print("⚠️  WARNING: Structure Filter DISABLED - --nsp3_model not provided!")
            print("!"*80)
            print("")
            print("   Running without structure filtering is NOT RECOMMENDED!")
            print("   Without structure filtering, generated binders may have:")
            print("   - Disordered terminal regions")
            print("   - Low secondary structure content")
            print("   - Non-compact, unstable folds")
            print("")
            print("   To enable structure filtering, obtain NetSurfP-3.0:")
            print("   ┌─────────────────────────────────────────────────────────────────┐")
            print("   │  https://services.healthtech.dtu.dk/services/NetSurfP-3.0/      │")
            print("   └─────────────────────────────────────────────────────────────────┘")
            print("   Request the official package, license, and model weights (nsp3.pth)")
            print("   Then run with: --nsp3_model /path/to/NetSurfP-3.0_standalone/models/nsp3.pth")
            print("")
            print("!"*80 + "\n")
            self.use_structure_filter = False
        
        # Prepare target sequence
        self._prepare_target()
    
    def _prepare_target(self):
        """Prepare target sequence from PDB or FASTA."""
        # If target_seq already provided, use it (don't extract from PDB)
        if self.target_seq is not None:
            print(f"Using provided target sequence: {len(self.target_seq)} residues")
            return
        else:
            raise ValueError(
                "No valid target input provided. Please provide:\n"
                "  --target_fasta <path_to_fasta>   : Use FASTA sequence"
            )
    
    def _get_worker_config(self, design_id: int, gpu_id: int) -> dict:
        """Build config dict for parallel worker."""
        return {
            'design_id': design_id,
            'gpu_id': gpu_id,  # Hint, actual device determined by worker identity
            'gpu_list': self.gpu_list,  # Full list for worker to choose from
            'accelerator_type': self.accelerator_type,  # 'cuda', 'mps', or 'cpu'
            'target_seq': self.target_seq,
            'gen_model_path': self.gen_model_path,
            'ppi_model_path': self.ppi_model_path,
            'nsp3_model_path': self.nsp3_model_path,
            'nsp3_esm_path': self.nsp3_esm_path,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'ppi_threshold': self.ppi_threshold,
            'ss_preference': self.ss_preference,
            'min_structured_fraction': self.min_structured_fraction,
            'use_structure_filter': self.use_structure_filter,
            'max_mean_disorder': self.max_mean_disorder,
            'min_buried_fraction': self.min_buried_fraction,
            'generate_batch_size': self.generate_batch_size,
            'output_dir': self.output_dir,
        }
    
    def _write_result_to_file(self, result: dict, output_csv: str):
        """Write a single result to the output file (with flush for immediate write)."""
        # Helper function to format floats or return N/A
        def fmt_float(val, precision=3):
            if isinstance(val, float):
                return f"{val:.{precision}f}"
            return "N/A"
        
        line = (
            f"design_{result['id']}\t"
            f"{self.target_seq}\t"
            f"{result['seq_original']}\t"
            f"{result['seq_trimmed']}\t"
            f"{result['length']}\t"
            f"{fmt_float(result.get('helix_content'))}\t"
            f"{fmt_float(result.get('strand_content'))}\t"
            f"{fmt_float(result.get('structured_fraction'))}\t"
            f"{fmt_float(result.get('coil_content'))}\t"
            f"{result.get('ss_type', 'N/A')}\t"
            f"{fmt_float(result.get('mean_disorder'))}\t"
            f"{fmt_float(result.get('buried_fraction'))}\t"
            f"{result['max_aa_percent']:.2f}\t"
            f"{result['hydrophobic_pct']:.2f}\t"
            f"{result['aromatic_pct']:.2f}\t"
            f"{result['charged_pct']:.2f}\t"
            f"{result['entropy']:.4f}\t"
            f"{result['net_charge']}\t"
            f"{result['ppi_prob']:.4f}\n"
        )
        with open(output_csv, 'a') as fout:
            fout.write(line)
            fout.flush()  # Ensure immediate write to disk
    
    def generate_batch_sequences(self) -> List[str]:
        """
        Generate a batch of binder sequences using LLM.
        
        Returns:
            List of raw generated sequences
        """
        if self.chat_model is None:
            raise RuntimeError("Generation model not loaded")
        
        # Truncate target sequence if too long
        max_len = HuggingfaceEngine.MAX_TARGET_SEQ_LENGTH
        if len(self.target_seq) > max_len:
            truncated_seq = self.target_seq[:max_len]
        else:
            truncated_seq = self.target_seq
        
        query = f'[Generate Binder] Seq=<{truncated_seq}>'
        messages = [{"role": "user", "content": query}]
        
        # Generate batch of sequences
        responses = self.chat_model.batch_chat(messages, self.generate_batch_size)
        
        # Clean up responses
        sequences = []
        for response in responses:
            seq = response.replace('Seq=<', '').replace('>', '').strip()
            # Extract only valid amino acids
            clean_seq = ''.join(c for c in seq.upper() if c in VALID_AAS)
            if clean_seq:
                sequences.append(clean_seq)
        
        torch_gc()
        return sequences
    
    def process_batch_pipeline(self, sequences: List[str], start_id: int = 0) -> List[dict]:
        """
        Process a batch of sequences through the full pipeline.
        
        Workflow: QC1 → NetSurfP-3.0 (batch) → QC2 → PPI (batch)
        
        This is much more efficient than processing sequences one by one because:
        1. NetSurfP-3.0 uses batched ESM embeddings
        2. PPI predictor uses batched embeddings with shared target
        
        Args:
            sequences: List of raw generated sequences
            start_id: Starting design ID for this batch
            
        Returns:
            List of results for sequences that passed all filters
        """
        if not sequences:
            return []
        
        results = []
        
        # ===================================================================
        # Step 1: QC1 - Fast pre-filter (alphabet, length)
        # ===================================================================
        qc1_passed = []
        for i, seq in enumerate(sequences):
            qc_result = qc1_prefilter(seq, self.qc_config)
            if qc_result.passed_all:
                qc1_passed.append(seq)
        
        print(f"  QC1: {len(qc1_passed)}/{len(sequences)} passed")
        
        if not qc1_passed:
            return []
        
        # ===================================================================
        # Step 2: Structure Filter (NetSurfP-3.0) - Batch prediction
        # ===================================================================
        if self.structure_filter is not None:
            sf_results = self.structure_filter.filter_batch(qc1_passed, self.ss_preference)
            
            # Clear ESM-1b cache after structure prediction
            torch_gc()
            
            sf_passed_seqs = []
            sf_passed_results = []
            sf_passed_original = []
            
            for seq, sf_result in zip(qc1_passed, sf_results):
                if sf_result.passed:
                    sf_passed_seqs.append(sf_result.trimmed_sequence)
                    sf_passed_results.append(sf_result)
                    sf_passed_original.append(seq)
            
            print(f"  Structure Filter: {len(sf_passed_seqs)}/{len(qc1_passed)} passed")
            
            if not sf_passed_seqs:
                return []
        else:
            # No structure filter - all sequences pass
            sf_passed_seqs = qc1_passed
            sf_passed_results = [None] * len(qc1_passed)
            sf_passed_original = qc1_passed
        
        # ===================================================================
        # Step 3: QC2 - Post-trim quality checks
        # ===================================================================
        qc2_passed_seqs = []
        qc2_passed_results = []
        qc2_sf_results = []
        qc2_original = []
        
        for seq, sf_result, orig_seq in zip(sf_passed_seqs, sf_passed_results, sf_passed_original):
            qc_result = qc2_postfilter(seq, self.qc_config)
            if qc_result.passed_all:
                qc2_passed_seqs.append(seq)
                qc2_passed_results.append(qc_result)
                qc2_sf_results.append(sf_result)
                qc2_original.append(orig_seq)
        
        print(f"  QC2: {len(qc2_passed_seqs)}/{len(sf_passed_seqs)} passed")
        
        if not qc2_passed_seqs:
            return []
        
        # ===================================================================
        # Step 4: PPI Prediction - Batch prediction
        # ===================================================================
        if self.ppi_predictor is not None:
            # Predict PPI for trimmed sequences
            ppi_probs = self.ppi_predictor.predict_batch(self.target_seq, qc2_passed_seqs)
            
            # Clear ESM2 cache after PPI prediction
            torch_gc()
            
            for i, (seq, ppi_prob, qc_result, sf_result, orig_seq) in enumerate(
                zip(qc2_passed_seqs, ppi_probs, qc2_passed_results, qc2_sf_results, qc2_original)
            ):
                if ppi_prob >= self.ppi_threshold:
                    # Build result dict
                    result = {
                        'id': start_id + len(results),
                        'seq_original': orig_seq,
                        'seq_trimmed': seq,
                        'ppi_prob': ppi_prob,
                        'length': len(seq),
                        'max_aa_percent': qc_result.max_single_aa_percent,
                        'hydrophobic_pct': qc_result.hydrophobic_percent,
                        'aromatic_pct': qc_result.aromatic_percent,
                        'charged_pct': qc_result.charged_percent,
                        'entropy': qc_result.shannon_entropy,
                        'net_charge': qc_result.net_charge,
                    }
                    
                    # Add structure filter metrics if available
                    if sf_result is not None:
                        result.update({
                            'helix_content': sf_result.helix_content,
                            'strand_content': sf_result.strand_content,
                            'coil_content': sf_result.coil_content,
                            'ss_type': sf_result.ss_filter_passed,
                            'ss3': sf_result.ss3,
                            'n_term_trim': sf_result.n_term_trim,
                            'c_term_trim': sf_result.c_term_trim,
                            'mean_disorder': sf_result.mean_disorder,
                            'buried_fraction': sf_result.buried_fraction,
                            'structured_fraction': sf_result.structured_fraction,
                        })
                    
                    results.append(result)
            
            # Explicitly clean up prediction results to prevent memory accumulation
            del ppi_probs
            
            ppi_passed = len(results)
            print(f"  PPI: {ppi_passed}/{len(qc2_passed_seqs)} passed (threshold: {self.ppi_threshold})")
        else:
            # No PPI filter - should not happen in normal usage
            print("  Warning: No PPI predictor loaded")
        
        return results
    
    def run(self, num_designs: int = 10) -> List[dict]:
        """
        Run the pipeline to generate binder sequences.
        
        Parallel Batch Generation (multi-GPU):
        - Each GPU worker loads its own copy of LLM, NetSurfP-3.0, PPI models
        - Each worker does batch generation (8 seq/call) + batch filtering
        - Workers run in parallel using ProcessPoolExecutor
        
        Args:
            num_designs: Number of successful designs to generate
            
        Returns:
            List of successful design results
        """
        results = []
        
        # Output file
        output_csv = os.path.join(self.output_dir, 'llmbind_designed_binders.tsv')
        
        # Write header
        with open(output_csv, 'w') as fout:
            header = 'ID\ttarget_seq\tbinder_seq\tbinder_optimize_seq\tlength\t'
            header += 'helix_fraction\tstrand_fraction\tstructured_fraction\tcoil_fraction\tss_type\t'
            header += 'mean_disorder\tburied_fraction\t'
            header += 'max_aa_percent\thydrophobic_pct\taromatic_pct\tcharged_pct\tentropy\tnet_charge\t'
            header += 'PPI_prob\n'
            fout.write(header)
        
        print(f"\n{'='*80}")
        print(f"Starting Binder Design Pipeline")
        print(f"Target: {num_designs} successful designs")
        print(f"Batch size: {self.generate_batch_size} sequences per LLM call")
        if self.parallel_workers > 1:
            print(f"Parallel workers: {self.parallel_workers} (GPUs: {self.gpu_list})")
        else:
            print(f"Mode: Single worker on {self.accelerator_type.upper()}")
        print(f"{'='*80}")
        
        # Start timing
        start_time = time.time()
        
        try:
            # ===============================================================
            # Parallel Batch Generation
            # ===============================================================
            
            if self.parallel_workers > 1:
                # -----------------------------------------------------------
                # Multi-GPU Parallel Mode with Dynamic Work Distribution
                # -----------------------------------------------------------
                # All workers share a counter and stop when global target is reached
                # This ensures no GPU sits idle while others are still working
                
                from multiprocessing import Manager
                manager = Manager()
                shared_counter = manager.Value('i', 0)  # Shared count of successful designs
                shared_lock = manager.Lock()
                result_queue = manager.Queue()
                
                # Build worker configs - all workers target the global num_designs
                worker_configs = []
                for i in range(self.parallel_workers):
                    gpu_id = self.gpu_list[i % len(self.gpu_list)]
                    
                    config = {
                        'worker_id': i,
                        'gpu_id': gpu_id,
                        'accelerator_type': self.accelerator_type,
                        'num_to_generate': num_designs,  # Global target, not per-worker
                        'target_seq': self.target_seq,
                        'gen_model_path': self.gen_model_path,
                        'ppi_model_path': self.ppi_model_path,
                        'nsp3_model_path': self.nsp3_model_path,
                        'nsp3_esm_path': self.nsp3_esm_path,
                        'min_length': self.min_length,
                        'max_length': self.max_length,
                        'ppi_threshold': self.ppi_threshold,
                        'ss_preference': self.ss_preference,
                        'min_structured_fraction': self.min_structured_fraction,
                        'use_structure_filter': self.use_structure_filter,
                        'max_mean_disorder': self.max_mean_disorder,
                        'min_buried_fraction': self.min_buried_fraction,
                        'generate_batch_size': self.generate_batch_size,
                        # LLM generation parameters
                        'temperature': self.temperature,
                        'top_k': self.top_k,
                        'top_p': self.top_p,
                    }
                    # Pass shared objects for dynamic work distribution
                    worker_configs.append((config, shared_counter, shared_lock, result_queue))
                
                print(f"\nLaunching {self.parallel_workers} parallel workers (dynamic work distribution)...")
                print(f"All workers contribute to global target of {num_designs} designs\n")
                
                # Run workers in parallel using spawn context for proper CUDA isolation
                mp_context = multiprocessing.get_context('spawn')
                with ProcessPoolExecutor(max_workers=self.parallel_workers, mp_context=mp_context) as executor:
                    futures = {
                        executor.submit(_batch_generation_worker, config): config[0]['worker_id']
                        for config in worker_configs
                    }
                    
                    # Poll queue and write results as they arrive (incremental save)
                    written_ids = set()
                    worker_contributions = {}
                    workers_done = 0
                    
                    while workers_done < len(futures):
                        # Check for completed workers
                        for future in list(futures.keys()):
                            if future.done():
                                worker_id = futures[future]
                                if worker_id not in worker_contributions:
                                    try:
                                        worker_results = future.result()
                                        worker_contributions[worker_id] = len(worker_results)
                                        workers_done += 1
                                        print(f"[Worker {worker_id}] Contributed {len(worker_results)} designs")
                                    except Exception as e:
                                        print(f"[Worker {worker_id}] Error: {e}")
                                        worker_contributions[worker_id] = 0
                                        workers_done += 1
                        
                        # Collect and write any new results from queue
                        new_results = []
                        while not result_queue.empty():
                            try:
                                r = result_queue.get_nowait()
                                if r['id'] not in written_ids:
                                    new_results.append(r)
                                    written_ids.add(r['id'])
                            except Exception:  # Queue may be empty despite check
                                break
                        
                        # Write new results immediately
                        for r in sorted(new_results, key=lambda x: x['id']):
                            results.append(r)
                            self._write_result_to_file(r, output_csv)
                            print(f"  💾 Saved design {r['id']} (PPI: {r['ppi_prob']:.4f})")
                        
                        # Brief sleep to avoid busy-waiting
                        if workers_done < len(futures):
                            time.sleep(0.5)
                
                # Final collection of any remaining results
                while not result_queue.empty():
                    try:
                        r = result_queue.get_nowait()
                        if r['id'] not in written_ids:
                            results.append(r)
                            self._write_result_to_file(r, output_csv)
                            written_ids.add(r['id'])
                    except Exception:  # Queue may be empty despite check
                        break
                
                # Sort by ID to maintain order
                results.sort(key=lambda x: x['id'])
                
                # Trim to requested number (in case of race condition overshoot)
                results = results[:num_designs]
                
                # Print contribution summary
                print(f"\nWorker contributions: {worker_contributions}")
                print(f"Total collected: {len(results)} designs")
                
            else:
                # -----------------------------------------------------------
                # Single Worker Mode (original batch pipeline)
                # -----------------------------------------------------------
                total_generated = 0
                batch_count = 0
                
                while len(results) < num_designs:
                    batch_count += 1
                    print(f"\n--- Batch {batch_count} ---")
                    
                    # Generate batch of sequences
                    #print(f"Generating {self.generate_batch_size} sequences...")
                    raw_sequences = self.generate_batch_sequences()
                    total_generated += len(raw_sequences)
                    print(f"  Generated: {len(raw_sequences)} sequences")
                    
                    if not raw_sequences:
                        print("  Warning: No valid sequences generated, retrying...")
                        continue
                    
                    # Process through pipeline
                    batch_results = self.process_batch_pipeline(raw_sequences, start_id=len(results))
                    
                    # Save results
                    for result in batch_results:
                        if len(results) < num_designs:
                            result['id'] = len(results)
                            results.append(result)
                            self._write_result_to_file(result, output_csv)
                    
                    print(f"\n  Progress: {len(results)}/{num_designs} designs")
                    
                    # Safety check (increased limit for strict filters)
                    if batch_count > num_designs * 10000:
                        pass_rate = len(results) / max(total_generated, 1) * 100
                        print(f"\n⚠️  Hit max batch limit ({batch_count}). Pass rate: {pass_rate:.2f}%")
                        print(f"Consider: relaxing filters or adjusting expectations")
                        break
                        
            # ===============================================================
            # Summary
            # ===============================================================
            elapsed_time = time.time() - start_time
            print(f"\n{'='*80}")
            print("PIPELINE COMPLETE")
            print(f"{'='*80}")
            print(f"Successful designs: {len(results)}")
            print(f"Time elapsed: {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
            if len(results) > 0:
                print(f"Average time per design: {elapsed_time/len(results):.2f}s")
            print(f"Results saved to: {output_csv}")
            print(f"{'='*80}")
        
        except KeyboardInterrupt:
            elapsed_time = time.time() - start_time
            print(f"\n{'='*80}")
            print("Pipeline interrupted!")
            print(f"Successful designs so far: {len(results)}")
            print(f"Time elapsed: {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
            print(f"Results saved to: {output_csv}")
            print(f"{'='*80}")
        
        return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Binder Design Pipeline with Structure-Aware Filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python binder_pipeline.py --target_fasta target.fasta --gen_model ./sft_model \\
        --nsp3_model ./NetSurfP-3.0_standalone/models/nsp3.pth \\
        --output_dir output --num_designs 10
        """
    )
    
    # Target specification (at least one required: --target_fasta OR --target_pdb)
    parser.add_argument('--target_fasta', type=str, default=None,
                        help='Target FASTA file containing one sequence')
    parser.add_argument('--target_pdb', type=str, default=None,
                        help='Target PDB/CIF file (single chain). Sequence will be extracted automatically. '
                             'Residues will be reindexed to start from 1 if needed.')
    parser.add_argument('--gen_model', type=str, default=None,
                        help='Path to sequence generation LLM')
    parser.add_argument('--ppi_model', type=str, default=DEFAULT_PPI_MODEL,
                        help=f'Path to PPI classifier model (default: DEFAULT_PPI_MODEL)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--num_designs', type=int, default=10,
                        help='Number of successful designs to generate (default: 10)')
    parser.add_argument('--generate_batch_size', type=int, default=8,
                        help='Number of sequences to generate per LLM call (default: 8). '
                             'Higher values are more efficient but use more memory.')
    parser.add_argument('--min_length', type=int, default=50,
                        help='Minimum binder sequence length (default: 50)')
    parser.add_argument('--max_length', type=int, default=150,
                        help='Maximum binder sequence length (default: 150)')
    parser.add_argument('--ppi_threshold', type=float, default=0.3,
                        help='PPI probability threshold (default: 0.3)')
    
    # LLM Generation parameters (auto-tuned based on target length if not specified)
    parser.add_argument('--temperature', type=float, default=None,
                        help='LLM sampling temperature. Auto-tuned if not specified: '
                             '1.0 for short targets (<75 aa), 0.6 for longer targets.')
    parser.add_argument('--top_k', type=int, default=None,
                        help='LLM top-k sampling. Auto-tuned if not specified: '
                             '80 for short targets (<75 aa), 40 for longer targets.')
    parser.add_argument('--top_p', type=float, default=None,
                        help='LLM top-p (nucleus) sampling. Auto-tuned if not specified: '
                             '0.95 for short targets (<75 aa), 0.9 for longer targets.')
    
    # Structure Filter arguments
    parser.add_argument('--nsp3_model', type=str, default=None,
                        help='Path to NetSurfP-3.0 model file (nsp3.pth). '
                             'If not provided, structure filtering is skipped.')
    parser.add_argument('--nsp3_esm', type=str, default=None,
                        help='Path to ESM-1b weights for NetSurfP-3.0. '
                             'Default: ~/.cache/torch/hub/checkpoints/esm1b_t33_650M_UR50S.pt')

    parser.add_argument('--no_structure_filter', action='store_true', default=False,
                        help='Skip structure filtering even if nsp3_model is provided')
    parser.add_argument('--ss_preference', type=str, default=None, choices=['helix', 'strand'],
                        help='Secondary structure preference: helix (only helix-dominant), '
                             'strand (only strand-dominant), or omit to accept all including mixed '
                             '(default: None, accepts helix/strand/mixed)')
    parser.add_argument('--min_structured_fraction', type=float, default=0.6,
                        help='Minimum structured fraction (H+E) required (default: 0.6)')
    parser.add_argument('--max_mean_disorder', type=float, default=0.10,
                        help='Maximum mean disorder probability (default: 0.10)')
    parser.add_argument('--min_buried_fraction', type=float, default=0.25,
                        help='Minimum fraction of buried residues (default: 0.25)')

    # Parallel processing
    parser.add_argument('--parallel_workers', type=int, default=0,
                        help='Number of parallel workers for generation. '
                             'Default (0): auto-detect GPUs. Each worker loads models on its own GPU.')    
    args = parser.parse_args()
    
    # Auto-detect accelerators (CUDA GPUs or Apple MPS)
    accelerator_type, device_list = _get_available_accelerators()
    if accelerator_type == 'cuda':
        print(f"Detected {len(device_list)} CUDA GPU(s): {device_list}")
    elif accelerator_type == 'mps':
        print("Detected Apple MPS (Metal Performance Shaders)")
    else:
        print("No GPU/MPS detected. Running on CPU.")
    
    use_accelerator = accelerator_type in ('cuda', 'mps')
    
    # Set parallel workers based on available GPUs
    if accelerator_type == 'cuda':
        num_gpus = len(device_list)
        if args.parallel_workers == 0:
            parallel_workers = num_gpus  # Default: one worker per GPU
        else:
            parallel_workers = min(args.parallel_workers, num_gpus)
            if args.parallel_workers > num_gpus:
                print(f"⚠️  Requested {args.parallel_workers} workers but only {num_gpus} GPUs. Using {num_gpus}.")
    else:
        parallel_workers = 1  # MPS/CPU: single worker
        if args.parallel_workers > 1:
            print(f"⚠️  Parallel workers only supported with CUDA. Using single worker.")
    
    # Force batch_size=1 on CPU to avoid slow generation
    if accelerator_type == 'cpu':
        if args.generate_batch_size > 1:
            print(f"⚠️  CPU mode: forcing --generate_batch_size=1 (batch generation is slow on CPU)")
            args.generate_batch_size = 1
    
    # Validate: either target_fasta or target_pdb is required
    if args.target_fasta is None and args.target_pdb is None:
        parser.error("Either --target_fasta or --target_pdb is required")
    
    # If both provided, use PDB and warn user
    if args.target_fasta is not None and args.target_pdb is not None:
        print(f"⚠️  Both --target_fasta and --target_pdb provided. Using sequence from PDB/CIF, --target_fasta will be ignored.")
    
    # Parse target sequence from input
    target_seq = None
    reindexed_pdb = None  # Track temp file for cleanup
    
    if args.target_pdb is not None:
        # Extract sequence from PDB/CIF file (takes priority if both provided)
        if not os.path.exists(args.target_pdb):
            parser.error(f"PDB/CIF file not found: {args.target_pdb}")
        
        if not BIOPYTHON_AVAILABLE:
            parser.error("BioPython is required for PDB/CIF input. Install with: pip install biopython")
        
        print(f"Extracting sequence from PDB/CIF: {args.target_pdb}")
        try:
            # Save reindexed PDB to output directory (will be cleaned up after pipeline)
            os.makedirs(args.output_dir, exist_ok=True)
            reindexed_pdb = os.path.join(args.output_dir, "target_reindexed.pdb")
            target_seq, already_formatted = reindex_extractseq_pdbcif(args.target_pdb, reindexed_pdb)
            
            if already_formatted:
                print(f"Loaded target sequence from PDB/CIF: {len(target_seq)} residues (already indexed from 1)")
            else:
                print(f"Loaded target sequence from PDB/CIF: {len(target_seq)} residues (reindexed to start from 1)")
        except Exception as e:
            parser.error(f"Failed to parse PDB/CIF file: {e}")
    
    else:
        # Parse target sequence from FASTA
        if not os.path.exists(args.target_fasta):
            parser.error(f"FASTA file not found: {args.target_fasta}")
        
        # Parse FASTA file
        sequences = []
        current_seq = []
        with open(args.target_fasta, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                        current_seq = []
                elif line:
                    current_seq.append(line)
            if current_seq:
                sequences.append(''.join(current_seq))
        
        if len(sequences) == 0:
            parser.error(f"No sequences found in FASTA file: {args.target_fasta}")
        if len(sequences) > 1:
            print("Warning: Multiple sequences found in FASTA file. Using the first sequence.")
        
        target_seq = sequences[0]
        print(f"Loaded target sequence from FASTA: {len(target_seq)} residues")
    
    # =========================================================================
    # LLM generation parameters (with user override options for tuning)
    # =========================================================================
    temperature = args.temperature if args.temperature is not None else 0.6
    top_k = args.top_k if args.top_k is not None else 4
    top_p = args.top_p if args.top_p is not None else 0.9
    
    print(f"\nLLM Generation Parameters:")
    print(f"  temperature = {temperature} {'(user-specified)' if args.temperature is not None else '(default)'}")
    print(f"  top_k = {top_k} {'(user-specified)' if args.top_k is not None else '(default)'}")
    print(f"  top_p = {top_p} {'(user-specified)' if args.top_p is not None else '(default)'}")
    
    # Create and run pipeline
    pipeline = BinderDesignPipeline(
        target_seq=target_seq,
        gen_model_path=args.gen_model,
        ppi_model_path=args.ppi_model,
        output_dir=args.output_dir,
        ppi_threshold=args.ppi_threshold,
        min_length=args.min_length,
        max_length=args.max_length,
        # LLM generation parameters
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        # Structure Filter parameters
        nsp3_model_path=args.nsp3_model,
        nsp3_esm_path=args.nsp3_esm,
        ss_preference=args.ss_preference,
        use_structure_filter=not args.no_structure_filter,
        min_structured_fraction=args.min_structured_fraction,
        max_mean_disorder=args.max_mean_disorder,
        min_buried_fraction=args.min_buried_fraction,
        # Batch generation
        generate_batch_size=args.generate_batch_size,
        # Parallel processing (multi-GPU)
        parallel_workers=parallel_workers,
        gpu_list=device_list,
        accelerator_type=accelerator_type,
    )
    
    # Print mode info
    print(f"\nDevice: {accelerator_type.upper()}")
    print(f"Batch generation: {args.generate_batch_size} sequences per LLM call")
    if parallel_workers > 1:
        print(f"Parallel workers: {parallel_workers} (one per GPU)")
    
    # Run pipeline (includes structure prediction if enabled)
    results = pipeline.run(num_designs=args.num_designs)
    
    # Cleanup: remove temporary reindexed PDB file
    if reindexed_pdb is not None and os.path.exists(reindexed_pdb):
        os.remove(reindexed_pdb)


if __name__ == "__main__":    
    main()
