import os
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .utils import setup_logger

logger = setup_logger("AlphaReq.DataLoader")

class PExpDataLoader:
    """
    Robust Loader and Preprocessor for the PROMISE Extended (P-EXP) Dataset.
    Safely parses requirement specifications, cleans strings, handles nulls,
    and assigns ISO/IEC/IEEE 29148 normative modal weights.
    """

    CLASS_NAMES = {
        'F': 'Functional',
        'SE': 'Security',
        'US': 'Usability',
        'O': 'Operational',
        'PE': 'Performance',
        'LF': 'Look and Feel',
        'A': 'Availability',
        'MN': 'Maintainability',
        'SC': 'Scalability',
        'FT': 'Fault Tolerance',
        'L': 'Legal',
        'PO': 'Portability'
    }

    def __init__(self, data_path: str = "B:/Promise/Paper2/data/p_exp.csv"):
        self.data_path = os.path.abspath(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.load_data()

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"P-EXP dataset not found at expected path: {self.data_path}")

        raw_df = pd.read_csv(self.data_path)
        
        # Validate expected columns
        required_cols = {'Requirement', 'Type', 'File'}
        if not required_cols.issubset(raw_df.columns):
            raise ValueError(f"Dataset missing required columns. Expected {required_cols}, found {raw_df.columns}")

        # Data cleaning and type enforcement
        raw_df['Requirement'] = raw_df['Requirement'].fillna("").astype(str)
        raw_df['Clean_Requirement'] = raw_df['Requirement'].apply(self._clean_text)
        raw_df['Type'] = raw_df['Type'].fillna("F").astype(str).str.strip().str.upper()
        raw_df['File'] = pd.to_numeric(raw_df['File'], errors='coerce').fillna(1).astype(int)

        # Drop completely empty requirements
        raw_df = raw_df[raw_df['Clean_Requirement'].str.len() > 0].reset_index(drop=True)

        # Compute normative weights based on ISO 29148 / RFC 2119 prescriptiveness
        raw_df['Normative_Weight'] = raw_df['Clean_Requirement'].apply(self._compute_normative_weight)
        
        self.df = raw_df
        logger.info(f"Loaded {len(self.df)} valid requirements across {self.df['File'].nunique()} projects from {self.data_path}")
        return self.df

    @staticmethod
    def _clean_text(text: str) -> str:
        if not isinstance(text, str):
            text = str(text or "")
        # Remove excess whitespace, trailing escapes
        cleaned = re.sub(r'\s+', ' ', text).strip()
        cleaned = re.sub(r'[\r\n\t]', ' ', cleaned)
        return cleaned

    @staticmethod
    def _compute_normative_weight(text: str) -> float:
        if not text:
            return 0.5
        lower = text.lower()
        if re.search(r'\b(shall|must)\b', lower):
            return 1.0  # Mandatory obligation
        elif re.search(r'\b(should|will)\b', lower):
            return 0.6  # Strong recommendation
        elif re.search(r'\b(may|can)\b', lower):
            return 0.3  # Optional permission
        return 0.5     # Neutral default

    def get_project_ids(self) -> List[int]:
        if self.df is None:
            self.load_data()
        return sorted(self.df['File'].unique().tolist())

    def get_project(self, project_id: int) -> pd.DataFrame:
        if self.df is None:
            self.load_data()
        sub = self.df[self.df['File'] == project_id].copy().reset_index(drop=True)
        return sub

    def get_dataset_summary(self) -> Dict:
        if self.df is None:
            self.load_data()
        return {
            'total_requirements': len(self.df),
            'total_projects': int(self.df['File'].nunique()),
            'class_distribution': self.df['Type'].value_counts().to_dict(),
            'avg_reqs_per_project': float(self.df.groupby('File').size().mean()),
            'median_reqs_per_project': float(self.df.groupby('File').size().median()),
            'normative_obligation_count': int((self.df['Normative_Weight'] == 1.0).sum())
        }
