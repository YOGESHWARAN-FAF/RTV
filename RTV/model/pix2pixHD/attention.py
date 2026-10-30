import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np


# https://vaclavkosar.com/ml/cross-attention-in-transformer-architecture
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, channels, num_heads, height, width):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = channels // num_heads
        self.height = height
        self.width = width
        self.query_projections = nn.ModuleList(
            [nn.Linear(self.dim_per_head, self.dim_per_head) for _ in range(num_heads)])
        self.key_projections = nn.ModuleList(
            [nn.Linear(self.dim_per_head, self.dim_per_head) for _ in range(num_heads)])
        self.value_projections = nn.ModuleList(
            [nn.Linear(self.dim_per_head, self.dim_per_head) for _ in range(num_heads)])
        self.positional_encoding = nn.Parameter(torch.randn(height * width, channels))
        self.final_projection = nn.Linear(channels, channels)

    def forward(self, query_features, key_value_features):
        batch_size = query_features.size(0)
        # Reshape input to (batch, height*width, channels)
        query_features = query_features.view(batch_size, self.height * self.width, -1)
        key_value_features = key_value_features.view(batch_size, self.height * self.width, -1)

        # Add positional encoding to features
        query_features += self.positional_encoding
        key_value_features += self.positional_encoding

        query_splits = torch.split(query_features, self.dim_per_head, dim=2)
        key_value_splits = torch.split(key_value_features, self.dim_per_head, dim=2)

        all_heads = []
        for i in range(self.num_heads):
            query = self.query_projections[i](query_splits[i])
            key = self.key_projections[i](key_value_splits[i])
            value = self.value_projections[i](key_value_splits[i])

            attention_scores = torch.matmul(query, key.transpose(-2, -1))
            attention_scores = attention_scores / (self.dim_per_head ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)
            attended_features = torch.matmul(attention_weights, value)

            all_heads.append(attended_features)

        # Concatenate all heads and apply final projection
        combined_features = torch.cat(all_heads, dim=2)
        output_features = self.final_projection(combined_features)

        # Reshape output to (batch, channels, height, width)
        output_features = output_features.view(batch_size, -1, self.height, self.width)
        return output_features

class MultiHeadSelfAttention(MultiHeadCrossAttention):
    def __init__(self, channels, num_heads, height, width):
        super(MultiHeadSelfAttention, self).__init__(channels, num_heads, height, width)

    def forward(self, features):
        query_features, key_value_features = features, features
        batch_size = query_features.size(0)
        # Reshape input to (batch, height*width, channels)
        query_features = query_features.view(batch_size, self.height * self.width, -1)
        key_value_features = key_value_features.view(batch_size, self.height * self.width, -1)

        # Add positional encoding to features
        query_features += self.positional_encoding
        key_value_features += self.positional_encoding

        query_splits = torch.split(query_features, self.dim_per_head, dim=2)
        key_value_splits = torch.split(key_value_features, self.dim_per_head, dim=2)

        all_heads = []
        for i in range(self.num_heads):
            query = self.query_projections[i](query_splits[i])
            key = self.key_projections[i](key_value_splits[i])
            value = self.value_projections[i](key_value_splits[i])

            attention_scores = torch.matmul(query, key.transpose(-2, -1))
            attention_scores = attention_scores / (self.dim_per_head ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)
            attended_features = torch.matmul(attention_weights, value)

            all_heads.append(attended_features)

        # Concatenate all heads and apply final projection
        combined_features = torch.cat(all_heads, dim=2)
        output_features = self.final_projection(combined_features)

        # Reshape output to (batch, channels, height, width)
        output_features = output_features.view(batch_size, -1, self.height, self.width)
        return output_features
