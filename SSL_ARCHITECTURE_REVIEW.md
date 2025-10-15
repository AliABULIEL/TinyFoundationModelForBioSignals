# SSL Architecture & Debugging Guide

## Problem Summary
**Error:** `Shape mismatch: pred torch.Size([25, 2, 2048]), target torch.Size([25, 2, 1024])`

**Expected Flow:**
```
Input: [25, 2, 1024] (batch, channels, time)
    ↓
Masking: [25, 2, 1024] (mask 40% of patches)
    ↓
Encoder: [25, 8, 192] (batch, patches, d_model)
    ↓
Decoder: [25, 2, 1024] (batch, channels, time)
```

**Actual Output:** `[25, 2, 2048]` ← **16 patches × 128 = 2048 (WRONG!)**

---

## Root Cause Analysis

### Issue: Encoder outputs 16 patches instead of 8

**Calculation:**
- Input time: 1024 samples
- Patch size: 128 samples
- Expected patches: 1024 / 128 = **8 patches**
- Actual patches from encoder: **16 patches** (causing 2048 output)

**Why 16 patches?**

Two possibilities:

1. **TTM is creating patches per channel:**
   - Input channels: 2 (PPG + ECG)
   - Patches per channel: 8
   - Total patches: 2 × 8 = **16 patches**
   - This happens if `last_hidden_state` shape is `[B, C*P, D]` instead of `[B, C, P, D]`

2. **get_encoder_output() is incorrectly processing TTM output:**
   - TTM might output: `[B, C, P, D]` = `[25, 2, 8, 192]`
   - Current code does: `features.mean(dim=1)` → `[25, 8, 192]` ✓ CORRECT
   - But maybe TTM outputs: `[B, P*C, D]` = `[25, 16, 192]`
   - Then mean(dim=1) is wrong dimension!

---

## Complete SSL Pipeline Review

### 1. **Data Flow:**

```
VitalDB Windows [N, 2, 1024]
    ↓
DataLoader: batch of [B, 2, 1024]
    ↓
Masking: [B, 2, 1024] → [B, 2, 1024] + mask_bool
    ↓
Encoder.get_encoder_output([B, 2, 1024])
    ↓ Transpose to [B, 1024, 2] (TTM expects [batch, time, channels])
    ↓ TTM backbone
    ↓ Extract last_hidden_state
    ↓ Process to [B, P, D]
    ↓
[B, 8, 192] ← **SHOULD BE THIS**
    ↓
Decoder([B, 8, 192])
    ↓ Linear: [B, 8, 192] → [B, 8, 256] (C*patch_size = 2*128)
    ↓ Reshape: [B, 8, 256] → [B, 8, 2, 128]
    ↓ Permute: [B, 8, 2, 128] → [B, 2, 8, 128]
    ↓ Reshape: [B, 2, 8, 128] → [B, 2, 1024]
    ↓
[B, 2, 1024] ← **EXPECTED OUTPUT**
```

### 2. **Critical Code Sections:**

#### A. **Encoder: get_encoder_output() in ttm_adapter.py**

```python
def get_encoder_output(self, x: torch.Tensor) -> torch.Tensor:
    """Get encoder output for SSL pretraining."""
    if self.using_real_ttm:
        # TTM expects [batch, time, channels]
        if x.dim() == 3 and x.size(1) == self.input_channels:
            x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        
        backbone_output = self.backbone(x)
        
        # Extract last_hidden_state
        if hasattr(backbone_output, 'last_hidden_state'):
            features = backbone_output.last_hidden_state
            # Expected shape: [batch, channels, patches, hidden]
            # Or could be: [batch, patches, channels, hidden]
            # Or could be: [batch, patches*channels, hidden]
            
            # ⚠️ THIS IS THE CRITICAL PART TO DEBUG
            # Average over channels: [batch, patches, hidden]
            features = features.mean(dim=1)  # Assumes [B, C, P, D]
            
            return features
```

**ISSUE:** We don't know the actual shape of `backbone_output.last_hidden_state`!

#### B. **Decoder: ReconstructionHead1D in decoders.py**

```python
def forward(self, latents: torch.Tensor) -> torch.Tensor:
    """Reconstruct signal from latent patch representations."""
    B, P, D = latents.shape  # e.g., [25, 8, 192]
    
    # Project: [B, P, D] -> [B, P, C*patch_size]
    x = self.proj(latents)  # [25, 8, 256] if C=2, patch_size=128
    
    # Reshape: [B, P, C*patch_size] -> [B, P, C, patch_size]
    x = x.reshape(B, P, self.n_channels, self.patch_size)
    # [25, 8, 2, 128]
    
    # Permute: [B, P, C, patch_size] -> [B, C, P, patch_size]
    x = x.permute(0, 2, 1, 3)  # [25, 2, 8, 128]
    
    # Fold patches: [B, C, P, patch_size] -> [B, C, T]
    T = P * self.patch_size  # 8 * 128 = 1024
    x = x.reshape(B, self.n_channels, T)  # [25, 2, 1024]
    
    return x
```

**ISSUE:** If P=16 (wrong), then T = 16 * 128 = 2048 (ERROR!)

#### C. **Pretrainer: SSLTrainer._train_epoch() in pretrainer.py**

```python
# Encode: [B, C, T] -> [B, P, D]
latents = self.encoder.get_encoder_output(masked_inputs)

# Decode: [B, P, D] -> [B, C, T]
reconstructed = self.decoder(latents)

# Loss
msm_loss = self.msm_criterion(reconstructed, inputs, mask_bool)
```

---

## Debugging Steps

### Step 1: Check Git Status

```bash
cd /Users/aliab/Desktop/TinyFoundationModelForBioSignals
git status
git log -1
```

**Verify the latest commit is:** `0d407e4` (Fix SSL pretrainer...)

### Step 2: Add Debug Prints to get_encoder_output()

Edit `src/models/ttm_adapter.py`, find `get_encoder_output()` and add:

```python
def get_encoder_output(self, x: torch.Tensor) -> torch.Tensor:
    """Get encoder output for SSL pretraining."""
    if self.using_real_ttm:
        # TTM expects [batch, time, channels]
        if x.dim() == 3 and x.size(1) == self.input_channels:
            x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        
        print(f"[DEBUG] Input to TTM backbone: {x.shape}")
        
        backbone_output = self.backbone(x)
        
        if hasattr(backbone_output, 'last_hidden_state'):
            features = backbone_output.last_hidden_state
            
            # ⚠️ CRITICAL DEBUG INFO
            print(f"[DEBUG] TTM last_hidden_state shape: {features.shape}")
            print(f"[DEBUG] Expected: [B={x.size(0)}, ?, P=8, D=192]")
            
            # Try to understand the structure
            if features.dim() == 3:
                print(f"[DEBUG] 3D tensor: [B={features.size(0)}, dim1={features.size(1)}, dim2={features.size(2)}]")
                print(f"[DEBUG] If dim1=patches: P={features.size(1)}, D={features.size(2)}")
            elif features.dim() == 4:
                print(f"[DEBUG] 4D tensor: [B={features.size(0)}, C={features.size(1)}, P={features.size(2)}, D={features.size(3)}]")
            
            # Average over channels: [batch, patches, hidden]
            if features.dim() == 4:
                # [B, C, P, D] -> [B, P, D]
                features = features.mean(dim=1)
            elif features.dim() == 3:
                # Already [B, P, D]? Or [B, C*P, D]?
                # Check if dim1 == expected_patches * num_channels
                expected_patches = self.num_patches
                if features.size(1) == expected_patches * self.input_channels:
                    print(f"[WARNING] TTM output has {features.size(1)} which is C*P, not P!")
                    print(f"[WARNING] This will cause 2x output size!")
            
            print(f"[DEBUG] Final encoder output shape: {features.shape}")
            print(f"[DEBUG] Expected: [B={x.size(0)}, P=8, D=192]")
            
            return features
```

### Step 3: Sync to Google Colab

If you're running on Colab and the repo is in Google Drive:

```python
# In Colab
!cd /content/drive/MyDrive/TinyFoundationModelForBioSignals && git pull
```

### Step 4: Run Training with Debug

```bash
python3 scripts/pretrain_vitaldb_ssl.py --mode fasttrack --epochs 1 --data-dir /path/to/data
```

Look for the `[DEBUG]` output to see what TTM is actually returning.

---

## Expected Solutions

### Solution 1: TTM outputs [B, C*P, D] not [B, C, P, D]

If TTM flattens channels and patches together:

```python
def get_encoder_output(self, x: torch.Tensor) -> torch.Tensor:
    # ... previous code ...
    
    if hasattr(backbone_output, 'last_hidden_state'):
        features = backbone_output.last_hidden_state
        
        # Check if features are [B, C*P, D]
        if features.dim() == 3:
            B, combined, D = features.shape
            expected_combined = self.num_patches * self.input_channels
            
            if combined == expected_combined:
                # Reshape [B, C*P, D] -> [B, C, P, D]
                features = features.reshape(B, self.input_channels, self.num_patches, D)
                # Average over channels: [B, C, P, D] -> [B, P, D]
                features = features.mean(dim=1)
        
        return features
```

### Solution 2: TTM outputs [B, P, C, D] not [B, C, P, D]

If TTM transposes patches and channels:

```python
def get_encoder_output(self, x: torch.Tensor) -> torch.Tensor:
    # ... previous code ...
    
    if hasattr(backbone_output, 'last_hidden_state'):
        features = backbone_output.last_hidden_state
        
        if features.dim() == 4:
            # Check order: [B, P, C, D] or [B, C, P, D]?
            if features.size(1) == self.num_patches:
                # It's [B, P, C, D] - permute to [B, C, P, D]
                features = features.permute(0, 2, 1, 3)
            
            # Now average over channels
            features = features.mean(dim=1)  # [B, P, D]
        
        return features
```

---

## Action Items

1. ✅ **Verify Git sync** - Make sure Colab has latest code
2. ⚠️ **Add debug prints** - Understand TTM's actual output shape
3. 🔧 **Fix get_encoder_output()** - Based on debug findings
4. ✅ **Test again** - Should output [25, 2, 1024] correctly

---

## Quick Reference: Expected Shapes

| Stage | Shape | Description |
|-------|-------|-------------|
| Input | [25, 2, 1024] | batch, channels, time |
| After masking | [25, 2, 1024] | same shape, some values masked |
| Transpose for TTM | [25, 1024, 2] | batch, time, channels |
| TTM last_hidden_state | **[?, ?, ?]** | ← NEED TO DEBUG THIS |
| Encoder output | [25, 8, 192] | batch, patches, d_model |
| Decoder input | [25, 8, 192] | batch, patches, d_model |
| Decoder output | [25, 2, 1024] | batch, channels, time |
| Target | [25, 2, 1024] | batch, channels, time |

**The fix is in get_encoder_output() - we need to see what TTM actually returns!**
