-- SCRIPT D'IMPORT SUPABASE - DATASET 460,012 LIGNES
-- Généré le 2025-08-26 07:01:13
-- Taille estimée: 11.4 MB

-- 1. CRÉER LE SCHÉMA (exécuter d'abord)
\i supabase_schema.sql

-- 2. IMPORTER LES DONNÉES
-- Attention: Adaptez les chemins selon votre configuration Supabase

\COPY aml.parties FROM 'parties.csv' CSV HEADER;
\COPY aml.accounts FROM 'accounts.csv' CSV HEADER; 
\COPY aml.transactions FROM 'transactions.csv' CSV HEADER;
\COPY aml.alerts FROM 'alerts.csv' CSV HEADER;

-- 3. VÉRIFICATIONS POST-IMPORT
SELECT 'IMPORT TERMINÉ - VÉRIFICATIONS' AS status;

-- Comptage des lignes
SELECT 
    'parties' AS table_name, COUNT(*) AS row_count 
FROM aml.parties
UNION ALL
SELECT 'accounts', COUNT(*) FROM aml.accounts  
UNION ALL
SELECT 'transactions', COUNT(*) FROM aml.transactions
UNION ALL
SELECT 'alerts', COUNT(*) FROM aml.alerts
ORDER BY row_count DESC;

-- Estimation de la taille
SELECT * FROM aml.vw_size_estimation;

-- 4. TESTS RAPIDES
SELECT 'Top 5 comptes par volume' AS test;
SELECT * FROM aml.vw_account_stats 
ORDER BY total_volume DESC LIMIT 5;

SELECT 'Activité par jour (10 premiers)' AS test;
SELECT * FROM aml.vw_daily_summary ORDER BY day LIMIT 10;

SELECT 'Connexions réseau (top 10)' AS test;
SELECT * FROM aml.vw_tx_network ORDER BY total_amount DESC LIMIT 10;

-- 5. STATISTIQUES FINALES
SELECT 
    '460,012 lignes importées' AS summary,
    '11.4 MB estimés' AS size,
    'Dataset prêt pour Finsight AI!' AS status;
