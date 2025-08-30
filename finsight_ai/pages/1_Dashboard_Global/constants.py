"""Constants for the Global Dashboard"""

# Bank color mapping (original corporate colors)
BANK_COLORS = {
    'BNP Paribas': '#00A651',
    'Société Générale': '#E4002B',
    'Crédit Agricole': '#00A651',
    'Banque Populaire': '#FF6600',
    "Caisse d'Épargne": '#E4002B',
    'Crédit Mutuel': '#003399',
    'La Banque Postale': '#FFD320',
    'HSBC France': '#DB0011',
    'LCL (Crédit Lyonnais)': '#0066CC',
    'Boursorama Banque': '#FF6600'
}

# Bank model_id to name mapping
BANK_MAPPING = {
    0: 'BNP Paribas',
    1: 'Société Générale',
    2: 'Crédit Agricole',
    3: 'Banque Populaire',
    4: "Caisse d'Épargne",
    5: 'Crédit Mutuel',
    6: 'La Banque Postale',
    7: 'HSBC France',
    8: 'LCL (Crédit Lyonnais)',
    9: 'Boursorama Banque'
}

# Transaction type mapping
TRANSACTION_TYPE_MAPPING = {
    'W': 'Withdrawal',
    'T': 'Transfer',
    'D': 'Deposit',
    'C': 'Cash',
    'R': 'Remittance'
}

# Bank logo filename mapping
BANK_LOGO_MAPPING = {
    'BNP Paribas': 'bnp.png',
    'Société Générale': 'societe_generale.png',
    'Crédit Agricole': 'credit_agricole.png',
    'Banque Populaire': 'banque_populaire.png',
    "Caisse d'Épargne": 'caisse_epargne.png',
    'Crédit Mutuel': 'credit_mutuel.png',
    'La Banque Postale': 'banque_postale.png',
    'HSBC France': 'hsbc.png',
    'LCL (Crédit Lyonnais)': 'lcl.png',
    'Boursorama Banque': 'boursorama.png'
}

# Database schema
SCHEMA_NAME = 'aml'

# Chart configuration
DEFAULT_CHART_HEIGHT = 520
BANK_CHART_HEIGHT = 600
KPI_CARD_HEIGHT = 120

# Cache TTL (seconds)
CACHE_TTL = 300 