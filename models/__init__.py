#!/usr/bin/python
# -*- coding:utf-8 -*-
from models.EFFGCN import EFFGCN_features
from models.AdversarialNet import AdversarialNet
from models.EFF_GCN import EFFGCN
from models.GCN import GCN
from models.RLControlRod import RLControlRodPenalty
from models.SoftEventEFFGCN import SoftEventEFFGCN_features
from models.GCN_features import GCN_features
try:
    from models.EFFGCN_noFission import EFFGCN_features_noFission
except ImportError:
    EFFGCN_features_noFission = None

try:
    from models.EFFGCN_noFusion import EFFGCN_features_noFusion
except ImportError:
    EFFGCN_features_noFusion = None
