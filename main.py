import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import requests
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import os
from langgraph.graph import Graph, StateGraph, END
from langchain.schema import BaseMessage
from collections import defaultdict, Counter
import numpy as np
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State definition for LangGraph
@dataclass
class AgentState:
    customer_id: str
    customer_profile: Dict[str, Any] = None
    purchase_patterns: Dict[str, Any] = None
    product_affinities: List[Dict[str, Any]] = None
    opportunity_scores: List[Dict[str, Any]] = None
    research_report: str = ""
    recommendations: List[Dict[str, Any]] = None
    messages: List[BaseMessage] = None

class GroqAPIClient:
    """Groq API client with error handling and retry mechanism."""
    
    def __init__(self, api_key: str):
        self.groq_api_key = api_key
        
    def call_groq_api(self, messages: list, max_retries: int = 3) -> str:
        """Call Groq API with robust error handling and retry mechanism."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": os.getenv("GROQ_MODEL", "deepseek-r1-distill-llama-70b"),
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.3
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except Exception as e:
                logger.warning(f"Groq API attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return f"Error calling Groq API: {e}"
        return "Failed to get response from Groq API"

class PostgreSQLManager:
    """PostgreSQL database manager for customer data."""
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or os.getenv(
            "DATABASE_URL", 
            "postgresql://postgres:devarajan#8@localhost:5432/customer_db"
        )
        self.connection = None
        self.max_retries = 3
        
    def create_connection(self):
        """Create PostgreSQL connection with reconnection logic."""
        for attempt in range(self.max_retries):
            try:
                self.connection = psycopg2.connect(
                    self.connection_string,
                    cursor_factory=RealDictCursor
                )
                self.create_tables()
                self.populate_sample_data()
                logger.info("PostgreSQL connection established")
                return True
            except Exception as e:
                logger.warning(f"PostgreSQL connection attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to connect to PostgreSQL after {self.max_retries} attempts")
                    return False
        return False
    
    def create_tables(self):
        """Create PostgreSQL tables."""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customer_purchases (
                id SERIAL PRIMARY KEY,
                customer_id VARCHAR(50) NOT NULL,
                product VARCHAR(100) NOT NULL,
                quantity INTEGER NOT NULL,
                unit_price DECIMAL(10,2) NOT NULL,
                total_price DECIMAL(10,2) NOT NULL,
                purchase_date DATE NOT NULL,
                customer_name VARCHAR(200) NOT NULL,
                industry VARCHAR(100) NOT NULL,
                annual_revenue BIGINT NOT NULL,
                number_of_employees INTEGER NOT NULL,
                customer_priority_rating VARCHAR(50),
                account_type VARCHAR(100),
                location VARCHAR(200),
                current_products TEXT,
                product_usage_percent INTEGER,
                cross_sell_synergy TEXT,
                last_activity_date DATE,
                opportunity_stage VARCHAR(100),
                opportunity_amount DECIMAL(12,2),
                opportunity_type VARCHAR(100),
                competitors TEXT,
                activity_status VARCHAR(50),
                activity_priority VARCHAR(50),
                activity_type VARCHAR(50),
                product_sku VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_customer_id ON customer_purchases(customer_id);
            CREATE INDEX IF NOT EXISTS idx_product ON customer_purchases(product);
            CREATE INDEX IF NOT EXISTS idx_purchase_date ON customer_purchases(purchase_date);
            CREATE INDEX IF NOT EXISTS idx_industry ON customer_purchases(industry);
        ''')
        
        self.connection.commit()
    
    def populate_sample_data(self):
        """Populate database with sample data."""
        cursor = self.connection.cursor()
        
        # Check if data already exists to avoid duplicates
        cursor.execute('SELECT COUNT(*) FROM customer_purchases')
        if cursor.fetchone()['count'] > 0:
            logger.info("Sample data already populated")
            return
        
        sample_data = [
            ('C001', 'Drill Bits', 9, 250.00, 2250.00, '2024-11-27', 'Edge Communications', 'Electronics', 139000000, 1000, 'Medium', 'Hot Customer - Direct', 'Austin, TX, USA', 'Core Management Platform', 100, 'Collaboration Suite, E', '2024-11-19', 'Closed Won', 75000.00, 'New Customer', 'John Deere, Mitsubishi', 'Completed', 'High', 'Call', 'GC1060'),
            ('C001', 'Protective Gloves', 9, 1000.00, 9000.00, '2024-10-23', 'Edge Communications', 'Electronics', 139000000, 1000, 'Medium', 'Hot Customer - Direct', 'Austin, TX, USA', 'Core Management Platform', 100, 'Collaboration Suite, E', '2024-11-19', 'Closed Won', 75000.00, 'New Customer', 'John Deere, Mitsubishi', 'Completed', 'High', 'Call', 'GC1060'),
            ('C001', 'Generators', 1, 1000.00, 1000.00, '2024-09-27', 'Edge Communications', 'Electronics', 139000000, 1000, 'Medium', 'Hot Customer - Direct', 'Austin, TX, USA', 'Core Management Platform', 100, 'Collaboration Suite, E', '2024-11-19', 'Closed Won', 75000.00, 'New Customer', 'John Deere, Mitsubishi', 'Completed', 'High', 'Call', 'GC1060'),
            ('C002', 'Drills', 4, 100.00, 400.00, '2024-06-10', 'Burlington Textiles Corp', 'Apparel', 350000000, 9000, 'High', 'Warm Customer - Direct', 'Burlington, NC, USA', 'Collaboration Suite', 85, 'Advanced Analytics, F', '2024-11-19', 'Closed Won', 235000.00, 'New Customer', 'John Deere', 'Completed', 'Medium', 'Email', 'GC1040'),
            ('C002', 'Backup Batteries', 9, 100.00, 900.00, '2024-06-15', 'Burlington Textiles Corp', 'Apparel', 350000000, 9000, 'High', 'Warm Customer - Direct', 'Burlington, NC, USA', 'Collaboration Suite', 85, 'Advanced Analytics, F', '2024-11-19', 'Closed Won', 235000.00, 'New Customer', 'John Deere', 'Completed', 'Medium', 'Email', 'GC1040'),
            ('C002', 'Workflow Automation', 6, 1000.00, 6000.00, '2025-05-18', 'Burlington Textiles Corp', 'Apparel', 350000000, 9000, 'High', 'Warm Customer - Direct', 'Burlington, NC, USA', 'Collaboration Suite', 85, 'Advanced Analytics, F', '2024-11-19', 'Closed Won', 235000.00, 'New Customer', 'John Deere', 'Completed', 'Medium', 'Email', 'GC1040'),
            ('C003', 'Protective Gloves', 2, 750.00, 1500.00, '2025-03-22', 'Pyramid Construction Inc.', 'Construction', 950000000, 2680, 'High', 'Warm Customer - Channel', 'Paris, France', 'Advanced Analytics', 70, 'Workflow Automation, G', '2024-11-19', 'Prospecting', 10000.00, 'Existing Customer - Upgrade', 'Caterpillar', 'Open', 'High', 'Call', 'GC5020'),
            ('C004', 'Advanced Analytics', 8, 250.00, 2000.00, '2024-07-10', 'Grand Hotels & Resorts', 'Hospitality', 500000000, 5600, 'High', 'Warm Customer - Direct', 'Chicago, IL, USA', 'Reporting Dashboard', 65, 'API Integrations, H', '2024-11-19', 'Closed Won', 210000.00, 'Existing Customer - Upgrade', 'Fujitsu', 'Completed', 'High', 'Email', 'GC3040'),
            ('C004', 'Safety Gear', 10, 1000.00, 10000.00, '2024-11-24', 'Grand Hotels & Resorts', 'Hospitality', 500000000, 5600, 'High', 'Warm Customer - Direct', 'Chicago, IL, USA', 'Reporting Dashboard', 65, 'API Integrations, H', '2024-11-19', 'Closed Won', 210000.00, 'Existing Customer - Upgrade', 'Fujitsu', 'Completed', 'High', 'Email', 'GC3040'),
            ('C005', 'Generators', 5, 250.00, 1250.00, '2025-03-25', 'United Oil & Gas Corp.', 'Energy', 5600000000, 145000, 'High', 'Hot Customer - Direct', 'New York, NY, USA', 'Workflow Automation', 60, 'AI Insights Module, J', '2024-11-19', 'Negotiation/Review', 270000.00, 'Existing Customer - Upgrade', 'Caterpillar, Hawkpower', 'Open', 'Medium', 'Call', 'SL9080'),
            ('C006', 'Drill Bits', 5, 250.00, 1250.00, '2024-08-15', 'Tech Solutions Inc', 'Electronics', 80000000, 500, 'High', 'Warm Customer - Direct', 'San Francisco, CA, USA', 'Basic Platform', 80, 'Advanced Analytics, Workflow Automation', '2024-11-15', 'Qualified', 45000.00, 'Cross-sell', 'Competitor A', 'Open', 'Medium', 'Email', 'GC1060'),
            ('C006', 'Advanced Analytics', 3, 500.00, 1500.00, '2024-09-01', 'Tech Solutions Inc', 'Electronics', 80000000, 500, 'High', 'Warm Customer - Direct', 'San Francisco, CA, USA', 'Basic Platform', 80, 'Advanced Analytics, Workflow Automation', '2024-11-15', 'Qualified', 45000.00, 'Cross-sell', 'Competitor A', 'Open', 'Medium', 'Email', 'GC3040'),
            ('C006', 'Safety Gear', 7, 300.00, 2100.00, '2024-09-15', 'Tech Solutions Inc', 'Electronics', 80000000, 500, 'High', 'Warm Customer - Direct', 'San Francisco, CA, USA', 'Basic Platform', 80, 'Advanced Analytics, Workflow Automation', '2024-11-15', 'Qualified', 45000.00, 'Cross-sell', 'Competitor A', 'Open', 'Medium', 'Email', 'SG2030'),
            ('C007', 'Generators', 3, 800.00, 2400.00, '2024-07-20', 'BuildRight Construction', 'Construction', 450000000, 1200, 'High', 'Hot Customer - Direct', 'Denver, CO, USA', 'Project Management', 90, 'Safety Gear, Protective Equipment', '2024-11-10', 'Closed Won', 120000.00, 'New Customer', 'Caterpillar, DeWalt', 'Completed', 'High', 'Call', 'GN4050'),
            ('C007', 'Safety Gear', 12, 150.00, 1800.00, '2024-08-05', 'BuildRight Construction', 'Construction', 450000000, 1200, 'High', 'Hot Customer - Direct', 'Denver, CO, USA', 'Project Management', 90, 'Safety Gear, Protective Equipment', '2024-11-10', 'Closed Won', 120000.00, 'New Customer', 'Caterpillar, DeWalt', 'Completed', 'High', 'Call', 'SG2030'),
            ('C007', 'Drill Bits', 15, 200.00, 3000.00, '2024-08-20', 'BuildRight Construction', 'Construction', 450000000, 1200, 'High', 'Hot Customer - Direct', 'Denver, CO, USA', 'Project Management', 90, 'Safety Gear, Protective Equipment', '2024-11-10', 'Closed Won', 120000.00, 'New Customer', 'Caterpillar, DeWalt', 'Completed', 'High', 'Call', 'GC1060'),
            ('C007', 'Backup Batteries', 6, 400.00, 2400.00, '2024-09-10', 'BuildRight Construction', 'Construction', 450000000, 1200, 'High', 'Hot Customer - Direct', 'Denver, CO, USA', 'Project Management', 90, 'Safety Gear, Protective Equipment', '2024-11-10', 'Closed Won', 120000.00, 'New Customer', 'Caterpillar, DeWalt', 'Completed', 'High', 'Call', 'BB3040'),
            ('C008', 'Advanced Analytics', 4, 600.00, 2400.00, '2024-06-25', 'Fashion Forward Ltd', 'Apparel', 280000000, 3500, 'Medium', 'Warm Customer - Channel', 'Milan, Italy', 'Inventory System', 75, 'Workflow Automation, Collaboration Suite', '2024-11-05', 'Proposal', 85000.00, 'Existing Customer - Upgrade', 'SAP, Oracle', 'Open', 'Medium', 'Email', 'AA5060'),
            ('C008', 'Workflow Automation', 2, 1200.00, 2400.00, '2024-07-15', 'Fashion Forward Ltd', 'Apparel', 280000000, 3500, 'Medium', 'Warm Customer - Channel', 'Milan, Italy', 'Inventory System', 75, 'Workflow Automation, Collaboration Suite', '2024-11-05', 'Proposal', 85000.00, 'Existing Customer - Upgrade', 'SAP, Oracle', 'Open', 'Medium', 'Email', 'WA6070'),
            ('C008', 'Protective Gloves', 10, 80.00, 800.00, '2024-08-01', 'Fashion Forward Ltd', 'Apparel', 280000000, 3500, 'Medium', 'Warm Customer - Channel', 'Milan, Italy', 'Inventory System', 75, 'Workflow Automation, Collaboration Suite', '2024-11-05', 'Proposal', 85000.00, 'Existing Customer - Upgrade', 'SAP, Oracle', 'Open', 'Medium', 'Email', 'PG7080'),
            ('C009', 'API Integrations', 5, 800.00, 4000.00, '2024-05-30', 'Luxury Resorts International', 'Hospitality', 750000000, 8500, 'High', 'Hot Customer - Direct', 'Miami, FL, USA', 'Guest Management', 85, 'Collaboration Suite, Reporting Dashboard', '2024-10-28', 'Closed Won', 180000.00, 'Existing Customer - Upgrade', 'Salesforce, Microsoft', 'Completed', 'High', 'Call', 'API8090'),
            ('C009', 'Collaboration Suite', 8, 450.00, 3600.00, '2024-06-18', 'Luxury Resorts International', 'Hospitality', 750000000, 8500, 'High', 'Hot Customer - Direct', 'Miami, FL, USA', 'Guest Management', 85, 'Collaboration Suite, Reporting Dashboard', '2024-10-28', 'Closed Won', 180000.00, 'Existing Customer - Upgrade', 'Salesforce, Microsoft', 'Completed', 'High', 'Call', 'CS9100'),
            ('C009', 'Safety Gear', 20, 120.00, 2400.00, '2024-07-08', 'Luxury Resorts International', 'Hospitality', 750000000, 8500, 'High', 'Hot Customer - Direct', 'Miami, FL, USA', 'Guest Management', 85, 'Collaboration Suite, Reporting Dashboard', '2024-10-28', 'Closed Won', 180000.00, 'Existing Customer - Upgrade', 'Salesforce, Microsoft', 'Completed', 'High', 'Call', 'SG2030'),
            ('C010', 'Generators', 8, 1000.00, 8000.00, '2024-04-15', 'Global Energy Solutions', 'Energy', 8500000000, 25000, 'High', 'Hot Customer - Direct', 'Houston, TX, USA', 'Operations Platform', 95, 'Advanced Analytics, Workflow Automation', '2024-10-20', 'Closed Won', 320000.00, 'Existing Customer - Upgrade', 'GE, Siemens', 'Completed', 'High', 'Call', 'GN4050'),
            ('C010', 'Backup Batteries', 15, 600.00, 9000.00, '2024-05-10', 'Global Energy Solutions', 'Energy', 8500000000, 25000, 'High', 'Hot Customer - Direct', 'Houston, TX, USA', 'Operations Platform', 95, 'Advanced Analytics, Workflow Automation', '2024-10-20', 'Closed Won', 320000.00, 'Existing Customer - Upgrade', 'GE, Siemens', 'Completed', 'High', 'Call', 'BB3040'),
            ('C010', 'Safety Gear', 25, 200.00, 5000.00, '2024-05-25', 'Global Energy Solutions', 'Energy', 8500000000, 25000, 'High', 'Hot Customer - Direct', 'Houston, TX, USA', 'Operations Platform', 95, 'Advanced Analytics, Workflow Automation', '2024-10-20', 'Closed Won', 320000.00, 'Existing Customer - Upgrade', 'GE, Siemens', 'Completed', 'High', 'Call', 'SG2030'),
            ('C010', 'Advanced Analytics', 6, 900.00, 5400.00, '2024-06-12', 'Global Energy Solutions', 'Energy', 8500000000, 25000, 'High', 'Hot Customer - Direct', 'Houston, TX, USA', 'Operations Platform', 95, 'Advanced Analytics, Workflow Automation', '2024-10-20', 'Closed Won', 320000.00, 'Existing Customer - Upgrade', 'GE, Siemens', 'Completed', 'High', 'Call', 'AA5060')
        ]
        
        cursor.executemany('''
            INSERT INTO customer_purchases (
                customer_id, product, quantity, unit_price, total_price, purchase_date,
                customer_name, industry, annual_revenue, number_of_employees,
                customer_priority_rating, account_type, location, current_products,
                product_usage_percent, cross_sell_synergy, last_activity_date,
                opportunity_stage, opportunity_amount, opportunity_type, competitors,
                activity_status, activity_priority, activity_type, product_sku
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', sample_data)
        
        self.connection.commit()
        logger.info(f"Populated database with {len(sample_data)} sample records")
    
    def get_customer_data(self, customer_id: str) -> List[Dict]:
        """Fetch customer data from PostgreSQL."""
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT * FROM customer_purchases 
            WHERE customer_id = %s 
            ORDER BY purchase_date DESC
        ''', (customer_id,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_all_customers(self) -> List[str]:
        """Get list of all customer IDs from PostgreSQL."""
        cursor = self.connection.cursor()
        cursor.execute('SELECT DISTINCT customer_id FROM customer_purchases ORDER BY customer_id')
        return [row['customer_id'] for row in cursor.fetchall()]

class CustomerContextAgent:
    """Agent to extract customer profile from database."""
    
    def __init__(self, db_manager: PostgreSQLManager, groq_client: GroqAPIClient):
        self.db_manager = db_manager
        self.groq_client = groq_client
    
    def execute(self, state: AgentState) -> AgentState:
        """Extract customer profile."""
        logger.info(f"Extracting customer profile for {state.customer_id}")
        
        customer_data = self.db_manager.get_customer_data(state.customer_id)
        
        if not customer_data:
            state.customer_profile = {"error": "Customer not found"}
            return state
        
        first_record = customer_data[0]
        total_spent = sum(record['total_price'] for record in customer_data)
        products_purchased = list(set(record['product'] for record in customer_data))
        purchase_count = len(customer_data)
        
        recent_purchases = [
            record for record in customer_data 
            if record['purchase_date'] >= datetime(2024, 6, 1).date()
        ]
        
        state.customer_profile = {
            "customer_id": state.customer_id,
            "customer_name": first_record['customer_name'],
            "industry": first_record['industry'],
            "annual_revenue": float(first_record['annual_revenue']),
            "number_of_employees": first_record['number_of_employees'],
            "location": first_record['location'],
            "customer_priority_rating": first_record['customer_priority_rating'],
            "account_type": first_record['account_type'],
            "current_products": first_record['current_products'],
            "total_spent": float(total_spent),
            "products_purchased": products_purchased,
            "purchase_count": purchase_count,
            "recent_purchases": len(recent_purchases),
            "opportunity_stage": first_record['opportunity_stage'],
            "cross_sell_synergy": first_record['cross_sell_synergy']
        }
        
        logger.info(f"Customer profile extracted: {state.customer_profile['customer_name']}")
        return state

class PurchasePatternAnalysisAgent:
    """Agent to analyze purchase patterns and identify opportunities."""
    
    def __init__(self, db_manager: PostgreSQLManager, groq_client: GroqAPIClient):
        self.db_manager = db_manager
        self.groq_client = groq_client
    
    def execute(self, state: AgentState) -> AgentState:
        """Analyze purchase patterns."""
        logger.info(f"Analyzing purchase patterns for {state.customer_id}")
        
        customer_data = self.db_manager.get_customer_data(state.customer_id)
        product_counts = Counter(record['product'] for record in customer_data)
        product_spending = defaultdict(float)
        
        for record in customer_data:
            product_spending[record['product']] += float(record['total_price'])
        
        avg_purchase_count = np.mean(list(product_counts.values()))
        frequent_products = [
            product for product, count in product_counts.items() 
            if count >= avg_purchase_count
        ]
        
        all_products = set()
        for customer_id in self.db_manager.get_all_customers():
            customer_products = self.db_manager.get_customer_data(customer_id)
            all_products.update(record['product'] for record in customer_products)
        
        customer_products = set(record['product'] for record in customer_data)
        missing_products = list(all_products - customer_products)
        
        state.purchase_patterns = {
            "product_counts": dict(product_counts),
            "product_spending": dict(product_spending),
            "frequent_products": frequent_products,
            "missing_products": missing_products,
            "total_products": len(customer_products),
            "avg_purchase_frequency": avg_purchase_count
        }
        
        logger.info(f"Purchase patterns analyzed: {len(frequent_products)} frequent products")
        return state

class ProductAffinityAgent:
    """Agent to suggest related/co-purchased products."""
    
    def __init__(self, db_manager: PostgreSQLManager, groq_client: GroqAPIClient):
        self.db_manager = db_manager
        self.groq_client = groq_client
    
    def get_popular_products_by_industry(self, industry: str, exclude_products: List[str] = None) -> List[str]:
        """Get popular products for a specific industry as fallback."""
        industry_products = {
            'Electronics': ['Advanced Analytics', 'API Integrations', 'Workflow Automation', 'Backup Batteries', 'Generators', 'Safety Gear'],
            'Apparel': ['Workflow Automation', 'Advanced Analytics', 'Collaboration Suite', 'Protective Gloves', 'Safety Gear'],
            'Hospitality': ['API Integrations', 'Collaboration Suite', 'Reporting Dashboard', 'Advanced Analytics', 'Safety Gear'],
            'Construction': ['Generators', 'Safety Gear', 'Protective Gloves', 'Drill Bits', 'Drills', 'Backup Batteries'],
            'Energy': ['Generators', 'Safety Gear', 'Backup Batteries', 'Advanced Analytics', 'Workflow Automation']
        }
        
        exclude_products = exclude_products or []
        available_products = industry_products.get(industry, [])
        return [product for product in available_products if product not in exclude_products]
    
    def build_cross_industry_matrix(self, target_industry: str) -> dict:
        """Build co-purchase matrix focusing on industry patterns."""
        co_purchase_matrix = defaultdict(lambda: defaultdict(float))
        all_customers = self.db_manager.get_all_customers()
        
        for customer_id in all_customers:
            customer_data = self.db_manager.get_customer_data(customer_id)
            if not customer_data:
                continue
                
            customer_industry = customer_data[0]['industry']
            industry_weight = 2.0 if customer_industry == target_industry else 1.0
            customer_products = [record['product'] for record in customer_data]
            
            for i, product1 in enumerate(customer_products):
                for j, product2 in enumerate(customer_products):
                    if i != j:
                        co_purchase_matrix[product1][product2] += industry_weight
        
        return co_purchase_matrix
    
    def calculate_market_based_affinities(self, customer_profile: dict) -> List[tuple]:
        """Calculate affinities based on market patterns and business logic."""
        affinities = []
        current_products = customer_profile['products_purchased']
        industry = customer_profile['industry']
        revenue = customer_profile['annual_revenue']
        
        product_synergies = {
            'Drill Bits': ['Drills', 'Safety Gear', 'Protective Gloves'],
            'Drills': ['Drill Bits', 'Safety Gear', 'Backup Batteries'],
            'Protective Gloves': ['Safety Gear', 'Drill Bits', 'Generators'],
            'Advanced Analytics': ['Reporting Dashboard', 'API Integrations', 'Workflow Automation'],
            'Backup Batteries': ['Generators', 'Safety Gear', 'Drills'],
            'Generators': ['Backup Batteries', 'Safety Gear', 'Protective Gloves'],
            'Safety Gear': ['Protective Gloves', 'Generators', 'Drill Bits'],
            'API Integrations': ['Advanced Analytics', 'Workflow Automation', 'Collaboration Suite'],
            'Workflow Automation': ['Advanced Analytics', 'API Integrations', 'Collaboration Suite'],
            'Collaboration Suite': ['API Integrations', 'Workflow Automation', 'Reporting Dashboard'],
            'Reporting Dashboard': ['Advanced Analytics', 'Collaboration Suite', 'API Integrations']
        }
        
        synergy_scores = defaultdict(float)
        for owned_product in current_products:
            if owned_product in product_synergies:
                for related_product in product_synergies[owned_product]:
                    if related_product not in current_products:
                        synergy_scores[related_product] += 2.0
        
        industry_boosts = {
            'Electronics': {
                'Advanced Analytics': 1.5,
                'API Integrations': 1.5,
                'Workflow Automation': 1.3,
                'Backup Batteries': 1.2
            },
            'Construction': {
                'Generators': 1.8,
                'Safety Gear': 1.6,
                'Protective Gloves': 1.5,
                'Drill Bits': 1.4
            },
            'Energy': {
                'Generators': 2.0,
                'Backup Batteries': 1.7,
                'Safety Gear': 1.5,
                'Advanced Analytics': 1.3
            },
            'Apparel': {
                'Workflow Automation': 1.6,
                'Advanced Analytics': 1.4,
                'Collaboration Suite': 1.3,
                'Safety Gear': 1.2
            },
            'Hospitality': {
                'API Integrations': 1.5,
                'Collaboration Suite': 1.4,
                'Reporting Dashboard': 1.3,
                'Advanced Analytics': 1.2
            }
        }
        
        if industry in industry_boosts:
            for product, boost in industry_boosts[industry].items():
                if product not in current_products:
                    synergy_scores[product] += boost
        
        if revenue > 1000000000:
            high_value_products = ['Advanced Analytics', 'Workflow Automation', 'API Integrations']
            for product in high_value_products:
                if product not in current_products:
                    synergy_scores[product] += 1.0
        
        for product, score in synergy_scores.items():
            affinities.append((product, score))
        
        return sorted(affinities, key=lambda x: x[1], reverse=True)
    
    def execute(self, state: AgentState) -> AgentState:
        """Suggest related products based on affinity analysis."""
        logger.info(f"Analyzing product affinities for {state.customer_id}")
        
        customer_products = state.customer_profile['products_purchased']
        logger.info(f"Customer current products: {customer_products}")
        
        co_purchase_matrix = self.build_cross_industry_matrix(state.customer_profile['industry'])
        affinity_scores = defaultdict(float)
        
        for owned_product in customer_products:
            if owned_product in co_purchase_matrix:
                for related_product, count in co_purchase_matrix[owned_product].items():
                    if related_product not in customer_products:
                        affinity_scores[related_product] += count
        
        logger.info(f"Traditional affinity scores found: {len(affinity_scores)}")
        
        market_affinities = self.calculate_market_based_affinities(state.customer_profile)
        logger.info(f"Market-based affinity scores found: {len(market_affinities)}")
        
        for product, score in market_affinities:
            affinity_scores[product] += score
        
        if not affinity_scores:
            logger.info("No affinities found, using industry fallback")
            popular_products = self.get_popular_products_by_industry(
                state.customer_profile['industry'], 
                customer_products
            )
            for i, product in enumerate(popular_products[:8]):
                affinity_scores[product] = 3.0 - (i * 0.3)
        
        sorted_affinities = sorted(
            affinity_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        logger.info(f"Final affinity scores: {len(sorted_affinities)} recommendations")
        
        reasoning = "Business logic and market analysis based recommendations."
        
        if sorted_affinities:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a product recommendation expert. Provide business reasoning for product recommendations."
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Customer Profile:
                        - Industry: {state.customer_profile['industry']}
                        - Current Products: {customer_products}
                        - Annual Revenue: ${state.customer_profile['annual_revenue']:,}
                        
                        Top recommended products with scores:
                        {[(product, f"{score:.1f}") for product, score in sorted_affinities[:5]]}
                        
                        Provide brief business reasoning for why these products would be valuable for this customer.
                        Focus on operational synergies and industry-specific benefits.
                        """
                    }
                ]
                
                reasoning = self.groq_client.call_groq_api(messages)
            except Exception as e:
                logger.warning(f"Failed to get Groq reasoning: {e}")
                reasoning = f"Recommended based on industry patterns ({state.customer_profile['industry']}) and product synergies with current portfolio."
        
        state.product_affinities = [
            {
                "product": product,
                "affinity_score": score,
                "reasoning": reasoning
            }
            for product, score in sorted_affinities
        ]
        
        logger.info(f"Product affinities calculated: {len(state.product_affinities)} recommendations")
        return state

class OpportunityScoringAgent:
    """Agent to score cross-sell and upsell opportunities."""
    
    def __init__(self, db_manager: PostgreSQLManager, groq_client: GroqAPIClient):
        self.db_manager = db_manager
        self.groq_client = groq_client
    
    def execute(self, state: AgentState) -> AgentState:
        """Score opportunities based on multiple factors."""
        logger.info(f"Scoring opportunities for {state.customer_id}")
        
        opportunities = []
        industry_products = {
            'Construction': ['Generators', 'Safety Gear', 'Protective Gloves', 'Drill Bits'],
            'Electronics': ['Advanced Analytics', 'API Integrations', 'Workflow Automation'],
            'Energy': ['Generators', 'Safety Gear', 'Backup Batteries'],
            'Hospitality': ['API Integrations', 'Collaboration Suite', 'Reporting Dashboard'],
            'Apparel': ['Workflow Automation', 'Advanced Analytics', 'Collaboration Suite']
        }
        
        related_keywords = {
            'Drill Bits': ['Drills'],
            'Backup Batteries': ['Generators'],
            'Protective Gloves': ['Safety Gear'],
            'Advanced Analytics': ['Reporting Dashboard'],
            'API Integrations': ['Workflow Automation']
        }
        
        for affinity in state.product_affinities[:10]:
            product = affinity['product']
            base_score = affinity['affinity_score']
            industry_fit = 1.5 if product in industry_products.get(state.customer_profile['industry'], []) else 1.0
            revenue_factor = min(state.customer_profile['annual_revenue'] / 100000000, 2.0)
            final_score = base_score * industry_fit * revenue_factor
            
            opportunity_type = "cross-sell"
            customer_products = state.customer_profile['products_purchased']
            for existing_product in customer_products:
                if product in related_keywords.get(existing_product, []) or existing_product in related_keywords.get(product, []):
                    opportunity_type = "upsell"
                    break
            
            opportunities.append({
                "product": product,
                "opportunity_type": opportunity_type,
                "score": round(final_score, 2),
                "base_score": base_score,
                "industry_fit": industry_fit,
                "revenue_factor": round(revenue_factor, 2),
                "reasoning": affinity['reasoning']
            })
        
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        state.opportunity_scores = opportunities
        logger.info(f"Opportunities scored: {len(opportunities)} products")
        return state

class RecommendationReportAgent:
    """Agent to generate comprehensive research report."""
    
    def __init__(self, groq_client: GroqAPIClient):
        self.groq_client = groq_client
    
    def execute(self, state: AgentState) -> AgentState:
        """Generate detailed research report."""
        logger.info(f"Generating research report for {state.customer_id}")
        
        customer = state.customer_profile
        patterns = state.purchase_patterns
        top_opportunities = state.opportunity_scores[:5]
        
        messages = [
            {
                "role": "system",
                "content": """You are a senior business analyst specializing in customer insights and revenue optimization. 
                Generate comprehensive, professional research reports with actionable recommendations."""
            },
            {
                "role": "user",
                "content": f"""
                Generate a detailed research report for cross-sell and upsell opportunities.
                
                CUSTOMER PROFILE:
                - Company: {customer['customer_name']}
                - Industry: {customer['industry']}
                - Annual Revenue: ${customer['annual_revenue']:,}
                - Employees: {customer['number_of_employees']:,}
                - Location: {customer['location']}
                - Total Spent: ${customer['total_spent']:,}
                - Products Purchased: {len(customer['products_purchased'])}
                - Current Priority: {customer['customer_priority_rating']}
                
                PURCHASE ANALYSIS:
                - Frequent Products: {patterns['frequent_products']}
                - Missing Opportunities: {len(patterns['missing_products'])} products
                - Average Purchase Frequency: {patterns['avg_purchase_frequency']:.1f}
                
                TOP OPPORTUNITIES:
                {chr(10).join([f"- {opp['product']} ({opp['opportunity_type']}, Score: {opp['score']})" for opp in top_opportunities])}
                
                Create a professional report with these sections:
                1. Executive Summary
                2. Customer Overview
                3. Purchase Pattern Analysis
                4. Market Context & Industry Insights
                5. Recommended Opportunities (with rationale)
                6. Implementation Strategy
                7. Expected Outcomes
                
                Make it actionable and specific to this customer's profile.
                """
            }
        ]
        
        research_report = self.groq_client.call_groq_api(messages)
        recommendations = [
            {
                "product": opp['product'],
                "type": opp['opportunity_type'],
                "score": opp['score'],
                "rationale": f"Score: {opp['score']} (Base: {opp['base_score']}, Industry Fit: {opp['industry_fit']}x, Revenue Factor: {opp['revenue_factor']}x)",
                "priority": "High" if opp['score'] > 10 else "Medium" if opp['score'] > 5 else "Low"
            }
            for opp in top_opportunities
        ]
        
        state.research_report = research_report
        state.recommendations = recommendations
        logger.info("Research report generated successfully")
        return state

class CrossSellUpsellAgent:
    """Main LangGraph agent orchestrator."""
    
    def __init__(self, groq_api_key: str):
        self.groq_client = GroqAPIClient(groq_api_key)
        self.db_manager = PostgreSQLManager()
        
        self.customer_context_agent = CustomerContextAgent(self.db_manager, self.groq_client)
        self.purchase_pattern_agent = PurchasePatternAnalysisAgent(self.db_manager, self.groq_client)
        self.product_affinity_agent = ProductAffinityAgent(self.db_manager, self.groq_client)
        self.opportunity_scoring_agent = OpportunityScoringAgent(self.db_manager, self.groq_client)
        self.report_agent = RecommendationReportAgent(self.groq_client)
        
        if not self.db_manager.create_connection():
            raise Exception("Failed to initialize database connection")
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("extract_customer_context", self.customer_context_agent.execute)
        workflow.add_node("analyze_purchase_patterns", self.purchase_pattern_agent.execute)
        workflow.add_node("calculate_product_affinity", self.product_affinity_agent.execute)
        workflow.add_node("score_opportunities", self.opportunity_scoring_agent.execute)
        workflow.add_node("generate_report", self.report_agent.execute)
        
        workflow.set_entry_point("extract_customer_context")
        workflow.add_edge("extract_customer_context", "analyze_purchase_patterns")
        workflow.add_edge("analyze_purchase_patterns", "calculate_product_affinity")
        workflow.add_edge("calculate_product_affinity", "score_opportunities")
        workflow.add_edge("score_opportunities", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile()
    
    def get_recommendations(self, customer_id: str) -> Dict[str, Any]:
        """Main API method to get recommendations for a customer."""
        try:
            initial_state = AgentState(customer_id=customer_id)
            result = self.workflow.invoke(initial_state)
            return {
                "customer_id": customer_id,
                "customer_profile": result.get("customer_profile"),
                "research_report": result.get("research_report"),
                "recommendations": result.get("recommendations"),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing customer {customer_id}: {e}")
            return {
                "customer_id": customer_id,
                "error": str(e),
                "status": "error"
            }

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Cross-Sell/Upsell Recommendation System",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🚀 Cross-Sell/Upsell Recommendation System")
    st.markdown("*Powered by LangGraph and Groq AI*")
    
    with st.sidebar:
        st.header("Configuration")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.warning("Please set GROQ_API_KEY environment variable")
            st.stop()
    
    try:
        agent = CrossSellUpsellAgent(groq_api_key)
        st.success("✅ System initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {e}")
        st.stop()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Select Customer")
        customers = agent.db_manager.get_all_customers()
        
        if not customers:
            st.error("No customers found in database")
            st.stop()
        
        selected_customer = st.selectbox(
            "Choose a customer:",
            customers,
            help="Select a customer to generate recommendations"
        )
        
        if selected_customer:
            customer_data = agent.db_manager.get_customer_data(selected_customer)
            if customer_data:
                st.subheader("Customer Overview")
                first_record = customer_data[0]
                st.write(f"**Name:** {first_record['customer_name']}")
                st.write(f"**Industry:** {first_record['industry']}")
                st.write(f"**Revenue:** ${first_record['annual_revenue']:,}")
                st.write(f"**Employees:** {first_record['number_of_employees']:,}")
                st.write(f"**Location:** {first_record['location']}")
                
                st.subheader("Recent Purchases")
                products = list(set(record['product'] for record in customer_data))
                for product in products[:5]:
                    st.write(f"• {product}")
    
    with col2:
        st.header("Generate Recommendations")
        
        if st.button("🔍 Analyze Customer & Generate Report", type="primary"):
            if not selected_customer:
                st.error("Please select a customer first")
            else:
                with st.spinner("Analyzing customer data and generating recommendations..."):
                    progress_bar = st.progress(0)
                    progress_bar.progress(20)
                    st.info("📋 Extracting customer profile...")
                    progress_bar.progress(40)
                    st.info("📈 Analyzing purchase patterns...")
                    progress_bar.progress(60)
                    st.info("🔗 Calculating product affinities...")
                    progress_bar.progress(80)
                    st.info("🎯 Scoring opportunities...")
                    progress_bar.progress(90)
                    st.info("📝 Generating research report...")
                    
                    result = agent.get_recommendations(selected_customer)
                    progress_bar.progress(100)
                    
                    if result['status'] == 'success':
                        st.success("✅ Analysis completed successfully!")
                        
                        st.header("📊 Analysis Results")
                        with st.expander("👤 Customer Profile", expanded=True):
                            profile = result['customer_profile']
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Spent", f"${profile['total_spent']:,.0f}")
                                st.metric("Products Purchased", profile['purchase_count'])
                            
                            with col2:
                                st.metric("Annual Revenue", f"${profile['annual_revenue']:,.0f}")
                                st.metric("Employees", f"{profile['number_of_employees']:,}")
                            
                            with col3:
                                st.metric("Priority Rating", profile['customer_priority_rating'])
                                st.metric("Recent Purchases", profile['recent_purchases'])
                        
                        with st.expander("🎯 Top Recommendations", expanded=True):
                            if result['recommendations']:
                                for i, rec in enumerate(result['recommendations'][:5], 1):
                                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                                    
                                    with col1:
                                        st.write(f"**{i}. {rec['product']}**")
                                        st.caption(rec['rationale'])
                                    
                                    with col2:
                                        st.metric("Type", rec['type'].title())
                                    
                                    with col3:
                                        st.metric("Score", f"{rec['score']:.1f}")
                                    
                                    with col4:
                                        priority_color = {
                                            "High": "🔴",
                                            "Medium": "🟡", 
                                            "Low": "🟢"
                                        }
                                        st.metric("Priority", f"{priority_color.get(rec['priority'], '⚪')} {rec['priority']}")
                                    
                                    st.divider()
                            else:
                                st.warning("No recommendations available")
                        
                        with st.expander("📑 Detailed Research Report", expanded=False):
                            if result['research_report']:
                                st.markdown(result['research_report'])
                            else:
                                st.warning("Research report not available")
                        
                        st.header("💾 Export Results")
                        download_data = {
                            "customer_analysis": result,
                            "generated_at": datetime.now().isoformat(),
                            "system_version": "1.0"
                        }
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📄 Download Full Report (JSON)",
                                data=json.dumps(download_data, indent=2),
                                file_name=f"customer_analysis_{selected_customer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        
                        with col2:
                            if result['recommendations']:
                                df_recommendations = pd.DataFrame(result['recommendations'])
                                csv_data = df_recommendations.to_csv(index=False)
                                st.download_button(
                                    label="📊 Download Recommendations (CSV)",
                                    data=csv_data,
                                    file_name=f"recommendations_{selected_customer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                    else:
                        st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

# FastAPI Endpoints
api_app = FastAPI(
    title="Cross-Sell/Upsell Recommendation API",
    description="LangGraph-powered customer analysis and recommendation system",
    version="1.0.0"
)

@api_app.on_event("startup")
async def startup_event():
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise Exception("GROQ_API_KEY environment variable not set")
    api_app.state.agent = CrossSellUpsellAgent(groq_key)

@api_app.get("/recommendation")
async def get_recommendation(customer_id: str):
    """Get cross-sell/upsell recommendations for a customer."""
    if not hasattr(api_app.state, 'agent'):
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    if not customer_id or not re.match(r"^C\d+$", customer_id):
        raise HTTPException(status_code=400, detail="Valid customer_id (e.g., C001) is required")
    
    try:
        result = api_app.state.agent.get_recommendations(customer_id)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_app.get("/customers")
async def get_customers():
    """Get list of available customer IDs."""
    if not hasattr(api_app.state, 'agent'):
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        customers = api_app.state.agent.db_manager.get_all_customers()
        return {"customers": customers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch customers: {str(e)}")

@api_app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

def run_streamlit_app():
    """Run the Streamlit application."""
    main()

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    uvicorn.run(api_app, host=host, port=port)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        logger.info("Starting FastAPI server...")
        run_api_server()
    else:
        logger.info("Starting Streamlit app...")
        run_streamlit_app()