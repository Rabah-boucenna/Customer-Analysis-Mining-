"""
this file contains the main functions used by FP-Growth 
"""

import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations
import os
import webbrowser
import numpy as np


#Main Classes :

class FPNode:
    """Node in the FPTree"""
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None
    
    def increment(self, count):
        """Increment the count of this node"""
        self.count += count


class FPTree:
    """FPTree data structure for efficient pattern mining"""
    def __init__(self, transactions, min_support):
        self.min_support = min_support
        self.header_table = defaultdict(list)
        self.root = FPNode('root', 1, None)
        
        #Counting Frequencies
        item_counts = Counter()
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        
        # Filtering of support
        self.frequent_items = {
            item: count 
            for item, count in item_counts.items() 
            if count >= min_support
        }
        
        # Build the FP-Tree
        for transaction in transactions:
            filtered = [item for item in transaction if item in self.frequent_items]
            filtered.sort(key=lambda x: self.frequent_items[x], reverse=True)
            if filtered:
                self._insert(filtered, self.root, 1)
    
    def _insert(self, transaction, node, count):
        """inserting into the tree"""
        if not transaction:
            return
        
        first = transaction[0]
        if first in node.children:
            node.children[first].increment(count)
        else:
            new_node = FPNode(first, count, node)
            node.children[first] = new_node
            if self.header_table[first]:
                self.header_table[first][-1].next = new_node
            self.header_table[first].append(new_node)
        
        if len(transaction) > 1:
            self._insert(transaction[1:], node.children[first], count)
    
    def get_paths(self, item):
        """Get all paths ending with the given item"""
        paths = []
        if item not in self.header_table:
            return paths
        
        for node in self.header_table[item]:
            path = []
            parent = node.parent
            while parent.item != 'root':
                path.append(parent.item)
                parent = parent.parent
            if path:
                paths.append((path[::-1], node.count))
        return paths


#Main Algorithm : 

def fp_growth(transactions, min_support, prefix=[]):
    """
    Parameters:
    transactions : list of list
    min_support : int
    prefix : list

    Returns:
    dict : {pattern_tuple: support_count}
    """
    patterns = {}
    tree = FPTree(transactions, min_support)
    
    if not tree.frequent_items:
        return patterns
    
    for item in sorted(tree.frequent_items.keys(), key=lambda x: tree.frequent_items[x]):
        new_prefix = prefix + [item]
        support = tree.frequent_items[item]
        patterns[tuple(new_prefix)] = support
        
        conditional_patterns = tree.get_paths(item)
        if conditional_patterns:
            conditional_transactions = []
            for pattern, count in conditional_patterns:
                conditional_transactions.extend([pattern] * count)
            
            conditional_patterns_dict = fp_growth(
                conditional_transactions, 
                min_support, 
                new_prefix
            )
            patterns.update(conditional_patterns_dict)
    
    return patterns


def generate_rules(patterns, min_confidence, total_transactions):
    """
    Generate association rules from frequent patterns
    
    Parameters:
    patterns : dict
        Frequent patterns from fp_growth()
    min_confidence : float
        Minimum confidence threshold (0-1)
    total_transactions : int
        Total number of transactions
    Returns:
    list : List of rule dictionaries with keys:
           'antecedent', 'consequent', 'support', 'confidence', 'lift'
    """
    rules = []
    
    for pattern, support in patterns.items():
        if len(pattern) < 2:
            continue
        
        for i in range(1, len(pattern)):
            for antecedent in combinations(pattern, i):
                consequent = tuple(set(pattern) - set(antecedent))
                
                if consequent and antecedent in patterns:
                    confidence = support / patterns[antecedent]
                    
                    if confidence >= min_confidence:
                        consequent_support = patterns.get(consequent, 0)
                        if consequent_support > 0:
                            lift = confidence / (consequent_support / total_transactions)
                        else:
                            lift = 0
                        
                        rules.append({
                            'antecedent': list(antecedent),
                            'consequent': list(consequent),
                            'support': support,
                            'confidence': confidence,
                            'lift': lift
                        })
    
    return rules


#Data loading

def load_and_sample_data(filepath, sample_size=None, sample_percent=None, random_state=42):
    """
    Load data from CSV with optional sampling
    
    Parameters:
    filepath : str
    sample_size : int, optional
    sample_percent : float, optional
    random_state : int
    
    Returns:
    tuple : (DataFrame, original_size)
    """
    try:
        df = pd.read_csv(filepath)
        original_size = len(df)
        
        if sample_size:
            if sample_size < len(df):
                df = df.sample(n=sample_size, random_state=random_state)
                print(f" Sampled {sample_size:,} records from {original_size:,} total")
            else:
                print(f" Loaded all {original_size:,} records")
        elif sample_percent:
            n_samples = int(len(df) * sample_percent / 100)
            df = df.sample(n=n_samples, random_state=random_state)
            print(f" Sampled {n_samples:,} records ({sample_percent}%) from {original_size:,}")
        else:
            
            if len(df) > 20000:
                n_samples = 15000
                df = df.sample(n=n_samples, random_state=random_state)
                print(f" Auto-sampled {n_samples:,} from {original_size:,} (use sample_size=None to disable)")
            else:
                print(f" Loaded all {original_size:,} records")
        
        return df, original_size
        
    except FileNotFoundError:
        print(f" Error: File '{filepath}' not found!")
        return None, 0
    except Exception as e:
        print(f" Error loading data: {e}")
        return None, 0


def prepare_data(df, features):
    """
    Convert DataFrame to transaction format
    
    Parameters:
   
    df : DataFrame
        Input data
    features : list
        Column names to include as features
    
    Returns:
    list : List of transactions (each transaction is a list of items)
    """
    transactions = []
    for _, row in df.iterrows():
        transaction = []
        for feature in features:
            if feature in row and pd.notna(row[feature]):
                val = str(row[feature]).strip()
                transaction.append(f"{feature}={val}")
        if transaction:
            transactions.append(transaction)
    return transactions


# Exploring Data for FPGrowth


def explore_data(df, features):
    """
    
    Parameters:
    df : DataFrame
        Input data
    features : list
    """
    print("DATA EXPLORATION")
    
    print(f"\nDataset: {len(df):,} records")
    print(f"Features: {features}")
    
    print("\n Feature Distributions ")
    for feature in features:
        if feature in df.columns:
            unique_vals = df[feature].nunique()
            print(f"\n{feature.upper()}: {unique_vals} unique values")
            print(df[feature].value_counts().head(10))


def suggest_parameters(total_records, sample_size=None):
    """
    Suggest optimal min_support based on dataset size
    
    Parameters:
    total_records : int
        Total records in original dataset
    sample_size : int, optional
        Size of sample being analyzed
    Returns:
    int : Suggested minimum support value
    """
    actual_size = sample_size if sample_size else total_records
    
    # Rule of thumb: 0.5% to 2% of dataset
    min_support_low = max(5, int(actual_size * 0.005))
    min_support_high = max(10, int(actual_size * 0.02))
    
    print("PARAMETER RECOMMENDATIONS")
    print(f"\nFor {actual_size:,} records:")
    print(f"  Conservative (fewer patterns):  min_support >= {min_support_high}")
    print(f"  Balanced:                       min_support = {(min_support_low + min_support_high)//2}")
    print(f"  Aggressive (more patterns):     min_support = {min_support_low}")
    print(f"\nConfidence: 0.3-0.5 is typical (lower = more rules)")
    
    return (min_support_low + min_support_high) // 2


# generating the html : 


def generate_html_report(patterns, rules, total_records, analyzed_records, 
                        output_filename='fpgrowth_results.html',
                        template_file='template.html'):
    
    try:
       
        with open(template_file, 'r', encoding='utf-8') as f:
            template = f.read()
    except FileNotFoundError:
        
        template = get_default_template()
    
    
    patterns_html = ""
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:100]
    for pattern, support in sorted_patterns:
        pattern_str = ' → '.join(pattern)
        pct = (support / analyzed_records) * 100
        patterns_html += f"""
            <div class="pattern">
                <span class="strong">{pattern_str}</span>
                <br><small>Support: {support} ({pct:.1f}%)</small>
            </div>
"""
    
    
    rules_html = ""
    if rules:
        sorted_rules = sorted(rules, key=lambda x: x['lift'], reverse=True)[:100]
        for rule in sorted_rules:
            ant = ' ∧ '.join(rule['antecedent'])
            cons = ' ∧ '.join(rule['consequent'])
            rules_html += f"""
            <div class="rule">
                <div class="strong">{ant} → {cons}</div>
                <div class="metrics">
                    <span class="metric">Support: {rule['support']}</span>
                    <span class="metric">Confidence: {rule['confidence']:.1%}</span>
                    <span class="metric" style="background: #d4edda;">Lift: {rule['lift']:.2f}</span>
                </div>
            </div>
"""
    else:
        rules_html = "<p class='no-data'>No rules found. Try lowering min_confidence.</p>"
    
    # Replace placeholders
    html = template.replace('{{TOTAL_RECORDS}}', str(total_records))
    html = html.replace('{{TOTAL_PATTERNS}}', str(len(patterns)))
    html = html.replace('{{TOTAL_RULES}}', str(len(rules)))
    html = html.replace('{{PATTERNS_CONTENT}}', patterns_html)
    html = html.replace('{{RULES_CONTENT}}', rules_html)
    
    # Save HTML
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_filename


def get_default_template():
    """Return embedded HTML template if external file not found"""
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FP-Growth Analysis Results</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }
        
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 20px; 
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 60px 40px; 
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="40" fill="rgba(255,255,255,0.05)"/></svg>');
            background-size: 100px 100px;
        }
        
        .header h1 { 
            font-size: 3em; 
            font-weight: 700; 
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .header p { 
            font-size: 1.2em; 
            opacity: 0.95;
            position: relative;
            z-index: 1;
        }
        
        .content { padding: 50px 40px; }
        
        .stats { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 30px; 
            margin: 30px 0 50px 0;
        }
        
        .stat-card { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 40px 30px; 
            border-radius: 15px; 
            text-align: center;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        }
        
        .stat-value { 
            font-size: 3.5em; 
            font-weight: 700;
            line-height: 1;
        }
        
        .stat-label { 
            font-size: 1em; 
            text-transform: uppercase; 
            margin-top: 15px;
            letter-spacing: 2px;
            opacity: 0.95;
        }
        
        h2 { 
            color: #1e3c72; 
            margin: 60px 0 30px 0; 
            padding-bottom: 20px; 
            border-bottom: 4px solid #667eea;
            font-size: 2em;
            font-weight: 600;
            position: relative;
        }
        
        h2::after {
            content: '';
            position: absolute;
            bottom: -4px;
            left: 0;
            width: 100px;
            height: 4px;
            background: #764ba2;
        }
        
        .pattern { 
            padding: 25px 30px; 
            margin: 20px 0; 
            background: linear-gradient(135deg, #f0f4ff 0%, #e6edff 100%);
            border-left: 6px solid #667eea; 
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .pattern:hover {
            transform: translateX(5px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        }
        
        .rule { 
            padding: 25px 30px; 
            margin: 20px 0; 
            background: white;
            border: 2px solid #e0e7ff; 
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            transition: all 0.2s ease;
        }
        
        .rule:hover {
            border-color: #667eea;
            box-shadow: 0 6px 25px rgba(102, 126, 234, 0.2);
        }
        
        .metrics { 
            display: flex; 
            gap: 15px; 
            margin-top: 15px; 
            flex-wrap: wrap;
        }
        
        .metric { 
            padding: 10px 20px; 
            background: linear-gradient(135deg, #e6edff 0%, #d9e4ff 100%);
            border-radius: 25px; 
            font-size: 0.9em;
            font-weight: 500;
            color: #1e3c72;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .strong { 
            font-weight: 700; 
            color: #667eea;
            font-size: 1.1em;
        }
        
        .no-data { 
            text-align: center; 
            padding: 60px 40px; 
            color: #999; 
            font-style: italic;
            font-size: 1.1em;
        }
        
        @media (max-width: 768px) {
            body { padding: 20px 10px; }
            .content { padding: 30px 20px; }
            .header { padding: 40px 20px; }
            .header h1 { font-size: 2em; }
            .stat-value { font-size: 2.5em; }
            h2 { font-size: 1.5em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1> FP-Growth Analysis Results</h1>
            <p>Pattern Mining and Association Rules Discovery</p>
        </div>
        <div class="content">
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{{TOTAL_RECORDS}}</div>
                    <div class="stat-label">Total Records</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{TOTAL_PATTERNS}}</div>
                    <div class="stat-label">Patterns Found</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{TOTAL_RULES}}</div>
                    <div class="stat-label">Rules Generated</div>
                </div>
            </div>
            
            <h2> Frequent Patterns</h2>
            {{PATTERNS_CONTENT}}
            
            <h2> Association Rules</h2>
            {{RULES_CONTENT}}
        </div>
    </div>
</body>
</html>"""


#utility functions

def print_top_patterns(patterns, n=10, total_records=None):
    """Pretty print top N patterns"""
    print(f"TOP {n} FREQUENT PATTERNS")
    
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:n]
    for i, (pattern, support) in enumerate(sorted_patterns, 1):
        pattern_str = ' → '.join(pattern)
        if total_records:
            pct = (support / total_records) * 100
            print(f"{i:2d}. {pattern_str}")
            print(f"    Support: {support} ({pct:.1f}%)\n")
        else:
            print(f"{i:2d}. {pattern_str} (Support: {support})\n")


def print_top_rules(rules, n=10):
    """Pretty print top N rules"""
    if not rules:
        print("\n No rules found")
        return
    
    print("\n")
    print(f"TOP {n} ASSOCIATION RULES (by Lift)")
    
    sorted_rules = sorted(rules, key=lambda x: x['lift'], reverse=True)[:n]
    for i, rule in enumerate(sorted_rules, 1):
        ant = ' ∧ '.join(rule['antecedent'])
        cons = ' ∧ '.join(rule['consequent'])
        print(f"{i:2d}. IF {ant}")
        print(f"    THEN {cons}")
        print(f"    Support: {rule['support']}, Confidence: {rule['confidence']:.1%}, Lift: {rule['lift']:.2f}\n")


def filter_rules(rules, min_lift=None, min_confidence=None, max_antecedent_len=None):
    """Filter rules based on criteria"""
    filtered = rules
    
    if min_lift:
        filtered = [r for r in filtered if r['lift'] >= min_lift]
    
    if min_confidence:
        filtered = [r for r in filtered if r['confidence'] >= min_confidence]
    
    if max_antecedent_len:
        filtered = [r for r in filtered if len(r['antecedent']) <= max_antecedent_len]
    
    return filtered


def patterns_to_dataframe(patterns):
    """Convert patterns dict to DataFrame"""
    return pd.DataFrame([
        {
            'pattern': ' → '.join(pattern),
            'items': list(pattern),
            'length': len(pattern),
            'support': support
        }
        for pattern, support in patterns.items()
    ]).sort_values('support', ascending=False)


def rules_to_dataframe(rules):
    """Convert rules list to DataFrame"""
    return pd.DataFrame([
        {
            'antecedent': ' ∧ '.join(rule['antecedent']),
            'consequent': ' ∧ '.join(rule['consequent']),
            'support': rule['support'],
            'confidence': rule['confidence'],
            'lift': rule['lift']
        }
        for rule in rules
    ]).sort_values('lift', ascending=False)