CREATE SCHEMA IF NOT EXISTS aml;
SET search_path = aml, public;

CREATE TABLE parties (
    party_id        SMALLINT PRIMARY KEY,  
    name            VARCHAR(50),           
    type            CHAR(1) DEFAULT 'I',   
    country         CHAR(2) DEFAULT 'US',  
    business        CHAR(1),               
    risk_rating     SMALLINT DEFAULT 0     
);

CREATE TABLE accounts (
    account_id      INTEGER PRIMARY KEY,   
    party_id        SMALLINT REFERENCES parties(party_id),
    init_balance    DECIMAL(10,2),
    start_date      SMALLINT,              
    end_date        SMALLINT,
    country         CHAR(2) DEFAULT 'US',
    business        CHAR(1),
    suspicious      BOOLEAN DEFAULT FALSE,
    is_fraud        BOOLEAN DEFAULT FALSE,
    model_id        SMALLINT
);

CREATE TABLE transactions (
    transaction_id  INTEGER PRIMARY KEY,   
    account_id      INTEGER REFERENCES accounts(account_id),
    counter_party   INTEGER,               
    tx_type         CHAR(1),              
    amount          DECIMAL(8,2),          
    tx_step         SMALLINT,             
    is_suspicious   BOOLEAN DEFAULT FALSE  
);

CREATE TABLE alerts (
    alert_id        INTEGER PRIMARY KEY,
    account_id      INTEGER REFERENCES accounts(account_id),
    alert_type      SMALLINT,              
    alert_score     DECIMAL(4,3),         
    created_at      DATE,                  
    status          CHAR(1) DEFAULT 'A',   
    escalated       BOOLEAN DEFAULT FALSE
);


CREATE INDEX idx_tx_account ON transactions(account_id);
CREATE INDEX idx_tx_step ON transactions(tx_step);
CREATE INDEX idx_tx_amount ON transactions(amount) WHERE amount > 1000; -- Index partiel
CREATE INDEX idx_tx_suspicious ON transactions(account_id) WHERE is_suspicious = true;

CREATE INDEX idx_acc_party ON accounts(party_id);
CREATE INDEX idx_acc_suspicious ON accounts(account_id) WHERE suspicious = true;

CREATE INDEX idx_alert_account ON alerts(account_id);
CREATE INDEX idx_alert_active ON alerts(account_id) WHERE status = 'A';



CREATE VIEW vw_account_stats AS
SELECT 
    a.account_id,
    a.party_id,
    a.init_balance,
    a.suspicious,
    COUNT(t.transaction_id) AS tx_count,
    COALESCE(SUM(t.amount), 0) AS total_volume,
    COALESCE(AVG(t.amount), 0) AS avg_amount,
    COALESCE(MAX(t.amount), 0) AS max_amount,
    COUNT(al.alert_id) AS alert_count,
    CASE 
        WHEN COUNT(t.transaction_id) > 100 AND AVG(t.amount) > 5000 THEN 'HIGH'
        WHEN COUNT(t.transaction_id) > 50 OR AVG(t.amount) > 2000 THEN 'MEDIUM' 
        ELSE 'LOW'
    END AS risk_level
FROM accounts a
LEFT JOIN transactions t ON a.account_id = t.account_id
LEFT JOIN alerts al ON a.account_id = al.account_id
GROUP BY a.account_id, a.party_id, a.init_balance, a.suspicious;

CREATE VIEW vw_tx_network AS
SELECT 
    t.account_id AS from_account,
    t.counter_party AS to_account,
    COUNT(*) AS tx_count,
    SUM(t.amount) AS total_amount,
    AVG(t.amount) AS avg_amount,
    COUNT(*) FILTER (WHERE t.is_suspicious) AS suspicious_count
FROM transactions t
WHERE t.counter_party IS NOT NULL
GROUP BY t.account_id, t.counter_party
HAVING COUNT(*) >= 2; 

CREATE VIEW vw_daily_summary AS
SELECT 
    tx_step AS day,
    COUNT(*) AS tx_count,
    COUNT(DISTINCT account_id) AS active_accounts,
    SUM(amount) AS daily_volume,
    AVG(amount) AS avg_amount,
    COUNT(*) FILTER (WHERE is_suspicious) AS suspicious_count
FROM transactions
GROUP BY tx_step
ORDER BY tx_step;


CREATE OR REPLACE FUNCTION calculate_risk_score(acc_id INTEGER)
RETURNS DECIMAL(4,3) AS $$
DECLARE
    score DECIMAL(4,3) := 0;
    tx_count INTEGER;
    avg_amount DECIMAL;
    alert_count INTEGER;
BEGIN
    SELECT COUNT(*), AVG(amount) 
    INTO tx_count, avg_amount
    FROM transactions 
    WHERE account_id = acc_id;
    
    SELECT COUNT(*) 
    INTO alert_count
    FROM alerts 
    WHERE account_id = acc_id AND status = 'A';
    
    score := score + (tx_count::DECIMAL / 100);
    score := score + (avg_amount / 10000);
    score := score + (alert_count * 2);
    
    RETURN LEAST(score, 9.999);
END;
$$ LANGUAGE plpgsql;

CREATE VIEW vw_size_estimation AS
SELECT 
    'parties' AS table_name,
    COUNT(*) AS row_count,
    ROUND(COUNT(*) * 20 / 1024.0, 2) AS estimated_kb  
FROM parties
UNION ALL
SELECT 
    'accounts',
    COUNT(*),
    ROUND(COUNT(*) * 35 / 1024.0, 2)  
FROM accounts
UNION ALL
SELECT 
    'transactions',
    COUNT(*),
    ROUND(COUNT(*) * 25 / 1024.0, 2)  
FROM transactions
UNION ALL
SELECT 
    'alerts',
    COUNT(*),
    ROUND(COUNT(*) * 30 / 1024.0, 2)  
FROM alerts;


COMMENT ON SCHEMA aml IS 'AML schema optimized for Supabase Free Tier (500MB max)';
COMMENT ON TABLE transactions IS 'Main table - optimized for 300k-500k rows';
COMMENT ON VIEW vw_account_stats IS 'Precomputed view for account analysis'; 