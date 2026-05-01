-- X-ray Defect Detection Database Setup
-- Run these commands in MySQL

-- 1. Create database
CREATE DATABASE IF NOT EXISTS xray_defects;

-- 2. Use the database
USE xray_defects;

-- 3. Create defect_data table
CREATE TABLE IF NOT EXISTS defect_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    defect_no VARCHAR(255) NOT NULL,
    satellite VARCHAR(255) NOT NULL,
    component_name VARCHAR(255) NOT NULL,
    component_id VARCHAR(255) NOT NULL,
    defects_detected TEXT NOT NULL,
    date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_defect_no (defect_no),
    INDEX idx_date (date)
);

-- 4. Create defect_info table  
CREATE TABLE IF NOT EXISTS defect_info (
    id INT AUTO_INCREMENT PRIMARY KEY,
    defect_no VARCHAR(255) NOT NULL,
    defect_types TEXT NOT NULL,
    features TEXT NOT NULL,
    user_remarks TEXT,
    accept_reject VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_defect_no (defect_no),
    INDEX idx_status (accept_reject)
);

-- 5. Verify tables created
SHOW TABLES;

-- 6. Check table structure
DESCRIBE defect_data;
DESCRIBE defect_info;

-- Done! Tables are ready.
