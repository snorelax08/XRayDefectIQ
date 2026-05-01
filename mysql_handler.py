"""
MySQL Database Handler
"""

import mysql.connector
from mysql.connector import Error
import streamlit as st

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'ISRO$$weld',  
    'database': 'xray_defects'
}

def create_connection():
    """Create database connection"""
    try:
        connection = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        return connection
    except Error as e:
        st.error(f"DB connection error: {e}")
        return None

def create_table():
    """Create tables"""
    connection = create_connection()
    if not connection:
        return
    
    cursor = connection.cursor()
    
    try:
        # Defect data table with all columns
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS defect_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            defect_no VARCHAR(255) NOT NULL,
            satellite VARCHAR(255) NOT NULL,
            component_name VARCHAR(255) NOT NULL,
            component_id VARCHAR(255) NOT NULL,
            defects_detected TEXT NOT NULL,
            date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Defect info table with all columns
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS defect_info (
            id INT AUTO_INCREMENT PRIMARY KEY,
            defect_no VARCHAR(255) NOT NULL,
            defect_types TEXT NOT NULL,
            features TEXT NOT NULL,
            user_remarks TEXT,
            accept_reject VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        connection.commit()
    except Error as e:
        st.error(f"Table creation error: {e}")
    finally:
        cursor.close()
        connection.close()

def insert_data(defect_no, satellite, component_name, component_id, defects_detected, date):
    """Insert defect data"""
    connection = create_connection()
    if not connection:
        return False
    
    cursor = connection.cursor()
    query = """
    INSERT INTO defect_data 
    (defect_no, satellite, component_name, component_id, defects_detected, date)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    
    try:
        cursor.execute(query, (defect_no, satellite, component_name, 
                              component_id, defects_detected, date))
        connection.commit()
        return True
    except Error as e:
        st.error(f"Insert error: {e}")
        return False
    finally:
        cursor.close()
        connection.close()

def insert_defect_info(defect_no, defect_types, features, user_remarks, accept_reject):
    """Insert defect info"""
    connection = create_connection()
    if not connection:
        return False
    
    cursor = connection.cursor()
    query = """
    INSERT INTO defect_info 
    (defect_no, defect_types, features, user_remarks, accept_reject)
    VALUES (%s, %s, %s, %s, %s)
    """
    
    try:
        cursor.execute(query, (defect_no, defect_types, features, 
                              user_remarks, accept_reject))
        connection.commit()
        return True
    except Error as e:
        st.error(f"Insert error: {e}")
        return False
    finally:
        cursor.close()
        connection.close()

def fetch_all_entries():
    """Fetch all defect records"""
    connection = create_connection()
    if not connection:
        return []
    
    cursor = connection.cursor(dictionary=True)
    query = "SELECT * FROM defect_data ORDER BY id DESC"
    
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except Error as e:
        st.error(f"Fetch error: {e}")
        return []
    finally:
        cursor.close()
        connection.close()

def fetch_all_defect_info():
    """Fetch all defect info"""
    connection = create_connection()
    if not connection:
        return []
    
    cursor = connection.cursor(dictionary=True)
    query = "SELECT * FROM defect_info ORDER BY id DESC"
    
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except Error as e:
        st.error(f"Fetch error: {e}")
        return []
    finally:
        cursor.close()
        connection.close()

# Initialize tables
try:
    create_table()
except:
    pass
