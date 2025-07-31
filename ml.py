import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_prepare_data(file_path):
    """Load Excel file and prepare data for analysis"""
    print("Loading data...")
    
    # Try reading with different header rows in case headers aren't in row 1
    try:
        df = pd.read_excel(file_path, header=0)  # Row 1 as headers
        print("Reading with row 1 as headers...")
    except:
        try:
            df = pd.read_excel(file_path, header=1)  # Row 2 as headers
            print("Reading with row 2 as headers...")
        except:
            df = pd.read_excel(file_path, header=None)  # No headers
            print("Reading without headers - will use first row as data...")
    
    print(f"\nDataFrame shape: {df.shape} (rows x columns)")
    print(f"First few rows of data:")
    print(df.head(3))
    
    print(f"\nColumn names found:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. '{col}' (type: {type(col)})")
    
    # Check if we have unnamed columns (indicates headers might be wrong)
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        print(f"\nWarning: Found {len(unnamed_cols)} unnamed columns - headers might be in wrong row")
        
        # Try reading from different rows
        for header_row in [1, 2, 3]:
            try:
                test_df = pd.read_excel(file_path, header=header_row)
                unnamed_test = [col for col in test_df.columns if 'Unnamed' in str(col)]
                if len(unnamed_test) < len(unnamed_cols):
                    print(f"Row {header_row+1} looks like better headers, using that...")
                    df = test_df
                    break
            except:
                continue
    
    # Clean up column names (remove extra spaces, standardize)
    df.columns = df.columns.astype(str).str.strip()
    
    print(f"\nCleaned column names:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. '{col}'")
    
    # Show sample data from each column to help identify the right ones
    print(f"\nSample data from each column:")
    for col in df.columns[:10]:  # Show first 10 columns
        sample_vals = df[col].dropna().head(3).tolist()
        print(f"  '{col}': {sample_vals}")
    
    # Manual column selection if auto-detection fails
    print(f"\n" + "="*50)
    print("MANUAL COLUMN SELECTION")
    print("="*50)
    print("Please identify your columns by number from the list above:")
    
    try:
        account_col_num = int(input("Enter number for 'Account Name' column: ")) - 1
        territory_col_num = int(input("Enter number for 'Territory' column: ")) - 1
        biologics_col_num = int(input("Enter number for 'Biologics' column: ")) - 1
        cdmo_col_num = int(input("Enter number for 'CDMO' column: ")) - 1
        clinical_col_num = int(input("Enter number for 'Clinical Trials' column: ")) - 1
        regenerative_col_num = int(input("Enter number for 'Regenerative' column: ")) - 1
        cellgene_col_num = int(input("Enter number for 'Cell/Gene' column: ")) - 1
        qaqc_col_num = int(input("Enter number for 'QA/QC' column: ")) - 1
        strategic_col_num = int(input("Enter number for 'Strategic Fit' column: ")) - 1
        
        # Map to actual column names
        cols_list = list(df.columns)
        classification_cols = [
            cols_list[biologics_col_num],
            cols_list[cdmo_col_num], 
            cols_list[clinical_col_num],
            cols_list[regenerative_col_num],
            cols_list[cellgene_col_num],
            cols_list[qaqc_col_num]
        ]
        strategic_col = cols_list[strategic_col_num]
        
        # Rename columns for easier reference
        df.rename(columns={
            cols_list[account_col_num]: 'account name',
            cols_list[territory_col_num]: 'territory'
        }, inplace=True)
        
        print(f"\nUsing these columns:")
        print(f"Account Name: '{cols_list[account_col_num]}'")
        print(f"Territory: '{cols_list[territory_col_num]}'")
        for i, col in enumerate(classification_cols):
            print(f"Classification {i+1}: '{col}'")
        print(f"Strategic Fit: '{strategic_col}'")
        
        # HANDLE DUPLICATE COMPANIES
        print(f"\n" + "="*50)
        print("DUPLICATE ANALYSIS")
        print("="*50)
        
        original_count = len(df)
        duplicates = df[df['account name'].duplicated(keep=False)]
        unique_dupes = duplicates['account name'].nunique()
        
        print(f"Total records: {original_count}")
        print(f"Duplicate company entries: {len(duplicates)} records")
        print(f"Companies with duplicates: {unique_dupes}")
        
        if len(duplicates) > 0:
            print(f"\nTop companies with most duplicate entries:")
            dupe_counts = duplicates['account name'].value_counts().head(10)
            for company, count in dupe_counts.items():
                print(f"  {company}: {count} entries")
            
            # Handle duplicates - aggregate by taking the highest classification
            classification_hierarchy = {'g': 4, 'p': 3, 'c': 2, 'n': 1, 'u': 0}
            strategic_hierarchy = {'very high': 5, 'high': 4, 'medium': 3, 'low': 2, 'very low': 1}
            
            def get_best_classification(series):
                """Get the highest ranking classification from a series"""
                series = series.str.lower().fillna('u')
                best_score = -1
                best_value = 'u'
                for val in series:
                    if val in classification_hierarchy and classification_hierarchy[val] > best_score:
                        best_score = classification_hierarchy[val]
                        best_value = val
                return best_value
            
            def get_best_strategic(series):
                """Get the highest strategic fit from a series"""
                series = series.str.lower().fillna('very low')
                best_score = -1
                best_value = 'very low'
                for val in series:
                    if val in strategic_hierarchy and strategic_hierarchy[val] > best_score:
                        best_score = strategic_hierarchy[val]
                        best_value = val
                return best_value
            
            print(f"\nDeduplicating by taking BEST classification for each company...")
            
            # Aggregate duplicates
            agg_dict = {}
            for col in classification_cols:
                agg_dict[col] = get_best_classification
            agg_dict[strategic_col] = get_best_strategic
            agg_dict['territory'] = 'first'  # Keep first territory
            
            df_deduped = df.groupby('account name').agg(agg_dict).reset_index()
            
            print(f"After deduplication: {len(df_deduped)} unique companies")
            print(f"Removed {original_count - len(df_deduped)} duplicate entries")
            
            # Keep both versions for comparison
            df_original = df.copy()
            df = df_deduped
        else:
            print("No duplicates found!")
            df_original = df.copy()
        
    except (ValueError, IndexError) as e:
        print(f"Error in column selection: {e}")
        print("Please run the script again and enter valid numbers.")
        return None, None, None, None
    
    # Create numerical mappings for analysis
    classification_mapping = {
        'g': 4,  # Global leader (highest value)
        'p': 3,  # Primary
        'c': 2,  # Secondary  
        'n': 1,  # Not involved
        'u': 0   # Unclear (lowest)
    }
    
    strategic_mapping = {
        'very high': 5,
        'high': 4,
        'medium': 3,
        'low': 2,
        'very low': 1
    }
    
    # Apply mappings to create score columns (case-insensitive)
    for col in classification_cols:
        df[f'{col}_score'] = df[col].str.lower().map(classification_mapping)
    
    if strategic_col:
        df['strategic_score'] = df[strategic_col].str.lower().map(strategic_mapping)
    
    # Calculate total investment score
    score_cols = [f'{col}_score' for col in classification_cols]
    if strategic_col:
        score_cols.append('strategic_score')
    
    # Only include columns that actually exist
    existing_score_cols = [col for col in score_cols if col in df.columns]
    
    if existing_score_cols:
        df['total_investment_score'] = df[existing_score_cols].sum(axis=1)
    else:
        print("Warning: No valid score columns found")
        df['total_investment_score'] = 0
    
    # Calculate additional metrics
    df['global_leader_count'] = sum((df[col].str.lower() == 'g').astype(int) for col in classification_cols)
    df['primary_count'] = sum((df[col].str.lower() == 'p').astype(int) for col in classification_cols)
    df['involvement_breadth'] = sum((df[col].str.lower().isin(['g', 'p', 'c'])).astype(int) for col in classification_cols)
    
    print(f"Data loaded successfully: {len(df)} unique companies")
    print(f"Found {len(classification_cols)} classification columns")
    
    return df, classification_cols, strategic_col, df_original if 'df_original' in locals() else df

def create_classification_distribution_heatmap(df, classification_cols):
    """Create heatmap showing distribution of classification levels"""
    print("\nCreating classification distribution heatmap...")
    
    # Count frequency of each classification level
    classification_summary = pd.DataFrame()
    
    for col in classification_cols:
        if col in df.columns:
            counts = df[col].value_counts()
            classification_summary[col] = counts
    
    # Fill missing values with 0 and reorder rows
    classification_summary = classification_summary.fillna(0)
    desired_order = ['g', 'p', 'c', 'n', 'u']
    classification_summary = classification_summary.reindex(desired_order, fill_value=0)
    
    # Create heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(classification_summary, 
                annot=True, fmt='g', cmap='RdYlGn',
                yticklabels=['Global Leader', 'Primary', 'Secondary', 'Not Involved', 'Unclear'])
    plt.title('Classification Distribution Across All Categories')
    plt.xlabel('Business Categories')
    plt.ylabel('Involvement Level')
    plt.tight_layout()
    plt.show()

def create_investment_score_heatmap(df, classification_cols, top_n=20):
    """Create heatmap of top companies by investment score"""
    print(f"\nCreating top {top_n} investment targets heatmap...")
    
    # Get score columns
    score_cols = [f'{col}_score' for col in classification_cols if col in df.columns]
    if 'strategic_score' in df.columns:
        score_cols.append('strategic_score')
    
    # Get top companies
    top_companies = df.nlargest(top_n, 'total_investment_score')
    
    if len(top_companies) == 0:
        print("No companies found with valid scores")
        return
    
    # Create heatmap of top companies
    plt.figure(figsize=(15, 10))
    heatmap_data = top_companies[score_cols].T
    
    # Use account name for columns, or index if account name not available
    if 'account name' in top_companies.columns:
        heatmap_data.columns = top_companies['account name']
    else:
        heatmap_data.columns = [f"Company_{i+1}" for i in range(len(top_companies))]
    
    # Create labels for y-axis
    y_labels = [col.replace('_score', '').title() for col in score_cols]
    
    sns.heatmap(heatmap_data, 
                annot=True, fmt='g', cmap='RdYlGn',
                yticklabels=y_labels)
    plt.title(f'Top {top_n} Investment Targets - Scored Heatmap')
    plt.xlabel('Company Names')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def create_territory_analysis(df, classification_cols):
    """Create territory-based analysis heatmap"""
    print("\nCreating territory analysis heatmap...")
    
    if 'territory' not in df.columns:
        print("Territory column not found, skipping territory analysis")
        return
    
    # Get score columns
    score_cols = [f'{col}_score' for col in classification_cols if col in df.columns]
    if 'strategic_score' in df.columns:
        score_cols.append('strategic_score')
    
    # Average scores by territory
    territory_analysis = df.groupby('territory')[score_cols].mean().round(2)
    
    if len(territory_analysis) == 0:
        print("No territory data available")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Create labels for x-axis
    x_labels = [col.replace('_score', '').title() for col in score_cols]
    
    sns.heatmap(territory_analysis, 
                annot=True, fmt='.2f', cmap='Blues',
                xticklabels=x_labels)
    plt.title('Average Investment Scores by Territory')
    plt.xlabel('Business Categories')
    plt.ylabel('Territory')
    plt.tight_layout()
    plt.show()

def create_global_leaders_analysis(df, classification_cols):
    """Analyze companies that are global leaders"""
    print("\nCreating global leaders analysis...")
    
    # Count global leadership positions
    gl_summary = pd.DataFrame()
    gl_companies = pd.DataFrame()
    
    for col in classification_cols:
        if col in df.columns:
            gl_count = (df[col] == 'g').sum()
            gl_summary[col] = [gl_count]
            
            # Get companies that are global leaders in this category
            leaders = df[df[col] == 'g']
            if len(leaders) > 0 and 'account name' in df.columns:
                print(f"\nGlobal Leaders in {col}:")
                for company in leaders['account name'].head(10):  # Show top 10
                    print(f"  - {company}")
    
    if not gl_summary.empty:
        plt.figure(figsize=(10, 4))
        sns.heatmap(gl_summary, annot=True, fmt='d', cmap='Reds')
        plt.title('Global Leaders Count by Category')
        plt.ylabel('Count')
        plt.show()
    
    # Find companies with multiple global leadership positions
    df['global_leader_count'] = 0
    for col in classification_cols:
        if col in df.columns:
            df['global_leader_count'] += (df[col] == 'g').astype(int)
    
    multi_leaders = df[df['global_leader_count'] > 1]
    if len(multi_leaders) > 0 and 'account name' in df.columns:
        print(f"\nCompanies with Multiple Global Leadership Positions:")
        for idx, row in multi_leaders.nlargest(10, 'global_leader_count').iterrows():
            print(f"  - {row['account name']}: {row['global_leader_count']} categories")

def create_strategic_fit_analysis(df):
    """Analyze strategic fit distribution"""
    print("\nCreating strategic fit analysis...")
    
    strategic_col = 'strategic fit for labware'
    # Find the actual strategic column name
    for col in df.columns:
        if 'strategic' in col.lower():
            strategic_col = col
            break
    
    if strategic_col not in df.columns:
        print("Strategic fit column not found, skipping analysis")
        return
    
    # Count strategic fit levels
    strategic_counts = df[strategic_col].value_counts()
    
    plt.figure(figsize=(10, 6))
    
    # Create bar plot for strategic fit
    plt.subplot(1, 2, 1)
    strategic_counts.plot(kind='bar', color='skyblue')
    plt.title('Strategic Fit Distribution')
    plt.xlabel('Strategic Fit Level')
    plt.ylabel('Number of Companies')
    plt.xticks(rotation=45)
    
    # Create pie chart
    plt.subplot(1, 2, 2)
    strategic_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Strategic Fit Distribution (%)')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.show()

def create_investment_tier_analysis(df, classification_cols):
    """Create investment tier classification and analysis"""
    print("\nCreating investment tier analysis...")
    
    # Define investment tiers based on multiple criteria
    def classify_investment_tier(row):
        global_leaders = sum(1 for col in classification_cols if row[col].lower() == 'g')
        primary_positions = sum(1 for col in classification_cols if row[col].lower() == 'p')
        total_score = row['total_investment_score']
        strategic_score = row.get('strategic_score', 0)
        
        if global_leaders >= 2 and strategic_score >= 4:
            return "Tier 1: Premium Targets"
        elif global_leaders >= 1 and strategic_score >= 3:
            return "Tier 2: High Priority"
        elif (global_leaders >= 1 or primary_positions >= 2) and strategic_score >= 2:
            return "Tier 3: Strong Candidates"
        elif primary_positions >= 1 or total_score >= 8:
            return "Tier 4: Emerging Opportunities"
        else:
            return "Tier 5: Low Priority"
    
    df['investment_tier'] = df.apply(classify_investment_tier, axis=1)
    
    # Create tier distribution visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Tier distribution pie chart
    tier_counts = df['investment_tier'].value_counts()
    axes[0,0].pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0,0].set_title('Investment Tier Distribution')
    
    # Tier vs Territory heatmap
    tier_territory = pd.crosstab(df['investment_tier'], df['territory'])
    sns.heatmap(tier_territory, annot=True, fmt='d', ax=axes[0,1], cmap='YlOrRd')
    axes[0,1].set_title('Investment Tiers by Territory')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Average scores by tier
    tier_scores = df.groupby('investment_tier')[['total_investment_score', 'global_leader_count', 'involvement_breadth']].mean()
    sns.heatmap(tier_scores.T, annot=True, fmt='.1f', ax=axes[1,0], cmap='RdYlGn')
    axes[1,0].set_title('Average Metrics by Investment Tier')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Strategic fit vs Investment score scatter
    tier_colors = {'Tier 1: Premium Targets': 'red', 'Tier 2: High Priority': 'orange', 
                   'Tier 3: Strong Candidates': 'yellow', 'Tier 4: Emerging Opportunities': 'lightblue',
                   'Tier 5: Low Priority': 'gray'}
    
    for tier in df['investment_tier'].unique():
        tier_data = df[df['investment_tier'] == tier]
        axes[1,1].scatter(tier_data['strategic_score'], tier_data['total_investment_score'], 
                         label=tier, alpha=0.6, c=tier_colors.get(tier, 'blue'))
    
    axes[1,1].set_xlabel('Strategic Fit Score')
    axes[1,1].set_ylabel('Total Investment Score')
    axes[1,1].set_title('Strategic Fit vs Investment Score by Tier')
    axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Print tier summaries
    print(f"\nINVESTMENT TIER BREAKDOWN:")
    for tier in tier_counts.index:
        count = tier_counts[tier]
        avg_score = df[df['investment_tier'] == tier]['total_investment_score'].mean()
        print(f"{tier}: {count} companies (avg score: {avg_score:.1f})")
        
        # Show top 3 companies in each tier
        top_companies = df[df['investment_tier'] == tier].nlargest(3, 'total_investment_score')
        for idx, row in top_companies.iterrows():
            print(f"  • {row['account name']} (Score: {row['total_investment_score']:.1f})")

def create_competitive_landscape_analysis(df, classification_cols):
    """Analyze competitive landscape by category"""
    print("\nCreating competitive landscape analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(classification_cols):
        # Count companies by classification level
        classification_counts = df[col].str.lower().value_counts()
        
        # Create stacked bar chart by territory
        territory_class = pd.crosstab(df['territory'], df[col].str.lower(), normalize='index') * 100
        territory_class.plot(kind='bar', stacked=True, ax=axes[i], 
                           colormap='RdYlGn', rot=45)
        axes[i].set_title(f'{col} - Competitive Landscape by Territory (%)')
        axes[i].set_ylabel('Percentage of Companies')
        axes[i].legend(title='Classification', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.show()
    
    # Market opportunity analysis
    print(f"\nMARKET OPPORTUNITY ANALYSIS:")
    for col in classification_cols:
        global_leaders = (df[col].str.lower() == 'g').sum()
        primary_players = (df[col].str.lower() == 'p').sum()
        total_active = (df[col].str.lower().isin(['g', 'p', 'c'])).sum()
        
        opportunity_score = max(0, 10 - (global_leaders * 3 + primary_players * 2))
        
        print(f"\n{col}:")
        print(f"  Global Leaders: {global_leaders}")
        print(f"  Primary Players: {primary_players}")
        print(f"  Total Active Companies: {total_active}")
        print(f"  Market Opportunity Score: {opportunity_score}/10 {'🔥' if opportunity_score >= 7 else '📈' if opportunity_score >= 4 else '⚠️'}")

def create_portfolio_optimization_analysis(df, classification_cols):
    """Create portfolio optimization recommendations"""
    print("\nCreating portfolio optimization analysis...")
    
    # Calculate portfolio diversification scores
    def calculate_diversification_score(companies_subset):
        """Calculate how well diversified a portfolio subset is"""
        diversity_score = 0
        for col in classification_cols:
            unique_levels = companies_subset[col].str.lower().nunique()
            diversity_score += unique_levels
        
        territory_diversity = companies_subset['territory'].nunique()
        return diversity_score + territory_diversity
    
    # Get top companies by different criteria
    top_by_score = df.nlargest(20, 'total_investment_score')
    top_global_leaders = df[df['global_leader_count'] >= 1].nlargest(20, 'global_leader_count')
    top_strategic_fit = df.nlargest(20, 'strategic_score')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Portfolio composition by investment tier
    portfolio_tiers = top_by_score['investment_tier'].value_counts()
    axes[0,0].pie(portfolio_tiers.values, labels=portfolio_tiers.index, autopct='%1.1f%%')
    axes[0,0].set_title('Top 20 Portfolio - Investment Tier Mix')
    
    # Geographic diversification
    portfolio_territories = top_by_score['territory'].value_counts()
    portfolio_territories.plot(kind='bar', ax=axes[0,1], color='skyblue')
    axes[0,1].set_title('Top 20 Portfolio - Geographic Distribution')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Category strength heatmap
    portfolio_strengths = pd.DataFrame()
    for col in classification_cols:
        portfolio_strengths[col] = top_by_score[col].str.lower().value_counts()
    portfolio_strengths = portfolio_strengths.fillna(0)
    
    sns.heatmap(portfolio_strengths, annot=True, fmt='g', ax=axes[1,0], cmap='Blues')
    axes[1,0].set_title('Top 20 Portfolio - Category Strengths')
    
    # Risk vs Reward scatter
    axes[1,1].scatter(df['involvement_breadth'], df['total_investment_score'], 
                     alpha=0.6, c=df['strategic_score'], cmap='viridis')
    axes[1,1].set_xlabel('Involvement Breadth (Risk Diversification)')
    axes[1,1].set_ylabel('Total Investment Score (Reward)')
    axes[1,1].set_title('Risk vs Reward Analysis')
    
    # Highlight top 20 companies
    axes[1,1].scatter(top_by_score['involvement_breadth'], top_by_score['total_investment_score'], 
                     color='red', s=100, alpha=0.7, label='Top 20 Portfolio')
    axes[1,1].legend()
    
    plt.colorbar(axes[1,1].collections[0], ax=axes[1,1], label='Strategic Fit Score')
    plt.tight_layout()
    plt.show()
    
    # Portfolio recommendations
    print(f"\nPORTFOLIO RECOMMENDATIONS:")
    print(f"Diversification Score - Top 20: {calculate_diversification_score(top_by_score)}")
    print(f"Average Investment Score: {top_by_score['total_investment_score'].mean():.2f}")
    print(f"Average Strategic Fit: {top_by_score['strategic_score'].mean():.2f}")
    
    print(f"\nTOP 5 MUST-HAVE COMPANIES:")
    for idx, row in top_by_score.head(5).iterrows():
        global_areas = [col for col in classification_cols if row[col].lower() == 'g']
        print(f"  {row['account name']} - Score: {row['total_investment_score']:.1f}")
        print(f"    Global leader in: {', '.join(global_areas) if global_areas else 'None'}")
        print(f"    Territory: {row['territory']}, Strategic Fit: {row.get('strategic_score', 'N/A')}")

def create_before_after_comparison(df_original, df_deduped):
    """Compare analysis before and after deduplication"""
    if len(df_original) == len(df_deduped):
        print("No duplicates were found, skipping comparison.")
        return
        
    print("\nCreating before/after deduplication comparison...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Score distribution comparison
    axes[0].hist(df_original['total_investment_score'], bins=20, alpha=0.7, label='With Duplicates', color='red')
    axes[0].hist(df_deduped['total_investment_score'], bins=20, alpha=0.7, label='Deduplicated', color='blue')
    axes[0].set_title('Investment Score Distribution')
    axes[0].set_xlabel('Total Investment Score')
    axes[0].set_ylabel('Number of Companies')
    axes[0].legend()
    
    # Territory comparison
    territory_orig = df_original['territory'].value_counts()
    territory_dedup = df_deduped['territory'].value_counts()
    
    x = range(len(territory_orig))
    width = 0.35
    axes[1].bar([i - width/2 for i in x], territory_orig.values, width, label='With Duplicates', alpha=0.7, color='red')
    axes[1].bar([i + width/2 for i in x], [territory_dedup.get(t, 0) for t in territory_orig.index], 
               width, label='Deduplicated', alpha=0.7, color='blue')
    axes[1].set_title('Territory Distribution Comparison')
    axes[1].set_xlabel('Territory')
    axes[1].set_ylabel('Number of Companies')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(territory_orig.index, rotation=45)
    axes[1].legend()
    
    # Global leaders comparison
    gl_orig = df_original['global_leader_count'].value_counts().sort_index()
    gl_dedup = df_deduped['global_leader_count'].value_counts().sort_index()
    
    x = range(max(len(gl_orig), len(gl_dedup)))
    axes[2].bar([i - width/2 for i in x], [gl_orig.get(i, 0) for i in x], width, 
               label='With Duplicates', alpha=0.7, color='red')
    axes[2].bar([i + width/2 for i in x], [gl_dedup.get(i, 0) for i in x], width, 
               label='Deduplicated', alpha=0.7, color='blue')
    axes[2].set_title('Global Leadership Distribution')
    axes[2].set_xlabel('Number of Global Leadership Positions')
    axes[2].set_ylabel('Number of Companies')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nDEDUPLICATION IMPACT:")
    print(f"Original entries: {len(df_original)}")
    print(f"After deduplication: {len(df_deduped)}")
    print(f"Entries removed: {len(df_original) - len(df_deduped)}")
    print(f"Data quality improvement: {((len(df_original) - len(df_deduped)) / len(df_original) * 100):.1f}% noise reduction")

def create_actionable_insights_report(df, classification_cols, strategic_col):
    """Generate actionable business insights"""
    print("\n" + "="*60)
    print("ACTIONABLE BUSINESS INSIGHTS REPORT")
    print("="*60)
    
    # Key metrics
    total_companies = len(df)
    tier1_companies = len(df[df['investment_tier'] == 'Tier 1: Premium Targets'])
    high_strategic_fit = len(df[df['strategic_score'] >= 4])
    
    print(f"\n📊 KEY METRICS:")
    print(f"Total companies analyzed: {total_companies}")
    print(f"Tier 1 premium targets: {tier1_companies} ({tier1_companies/total_companies*100:.1f}%)")
    print(f"High strategic fit companies: {high_strategic_fit} ({high_strategic_fit/total_companies*100:.1f}%)")
    
    # Market opportunities
    print(f"\n🎯 TOP MARKET OPPORTUNITIES:")
    market_opps = []
    for col in classification_cols:
        global_leaders = (df[col].str.lower() == 'g').sum()
        if global_leaders <= 3:  # Less crowded markets
            total_active = (df[col].str.lower().isin(['g', 'p'])).sum()
            opportunity_score = 10 - (global_leaders * 3)
            market_opps.append((col, opportunity_score, global_leaders))
    
    market_opps.sort(key=lambda x: x[1], reverse=True)
    for i, (market, score, leaders) in enumerate(market_opps[:3]):
        print(f"{i+1}. {market} - Opportunity Score: {score}/10 ({leaders} global leaders)")
    
    # Geographic insights
    print(f"\n🌍 GEOGRAPHIC INSIGHTS:")
    territory_scores = df.groupby('territory')['total_investment_score'].agg(['mean', 'count']).round(2)
    territory_scores = territory_scores.sort_values('mean', ascending=False)
    
    print("Average investment score by territory:")
    for territory, (avg_score, count) in territory_scores.iterrows():
        print(f"  {territory}: {avg_score} (from {count} companies)")
    
    # Immediate action items
    print(f"\n⚡ IMMEDIATE ACTION ITEMS:")
    
    # Priority targets
    priority_targets = df[
        (df['investment_tier'].isin(['Tier 1: Premium Targets', 'Tier 2: High Priority'])) &
        (df['strategic_score'] >= 4)
    ].nlargest(5, 'total_investment_score')
    
    print("1. PRIORITY OUTREACH TARGETS:")
    for idx, row in priority_targets.iterrows():
        print(f"   • {row['account name']} - {row['territory']}")
        print(f"     Score: {row['total_investment_score']:.1f}, Strategic Fit: {row.get(strategic_col, 'N/A')}")
    
    # Undervalued opportunities
    undervalued = df[
        (df['global_leader_count'] >= 1) & 
        (df['strategic_score'] <= 2)
    ].nlargest(3, 'global_leader_count')
    
    if len(undervalued) > 0:
        print("\n2. UNDERVALUED OPPORTUNITIES (Strong capabilities, low strategic fit):")
        for idx, row in undervalued.iterrows():
            print(f"   • {row['account name']} - Consider partnership/acquisition")
    
    # Market gaps
    print("\n3. MARKET GAPS TO EXPLOIT:")
    for col in classification_cols:
        no_leaders = (df[col].str.lower() == 'g').sum() == 0
        few_players = (df[col].str.lower().isin(['g', 'p'])).sum() <= 5
        if no_leaders or few_players:
            status = "No global leaders" if no_leaders else "Few competitors"
            print(f"   • {col}: {status} - High opportunity market")
    
    # Territory expansion
    weak_territories = territory_scores[territory_scores['mean'] < territory_scores['mean'].mean()]
    if len(weak_territories) > 0:
        print("\n4. TERRITORY EXPANSION OPPORTUNITIES:")
        for territory, (avg_score, count) in weak_territories.iterrows():
            if count >= 3:  # Only suggest if there are enough companies
                print(f"   • {territory}: Below-average engagement, {count} potential targets")

def export_detailed_results(df, file_path):
    """Export detailed results for further analysis"""
    print("\nExporting detailed results...")
    
    # Create summary sheet
    summary_data = {
        'Company': df['account name'],
        'Territory': df['territory'],
        'Investment_Tier': df['investment_tier'],
        'Total_Score': df['total_investment_score'],
        'Strategic_Fit_Score': df['strategic_score'],
        'Global_Leader_Count': df['global_leader_count'],
        'Primary_Count': df['primary_count'],
        'Involvement_Breadth': df['involvement_breadth']
    }
    
    # Add individual classification scores
    for col in df.columns:
        if col.endswith('_score') and col != 'total_investment_score' and col != 'strategic_score':
            category = col.replace('_score', '')
            summary_data[f'{category}_Score'] = df[col]
    
    summary_df = pd.DataFrame(summary_data)
    
    # Export multiple sheets
    output_file = file_path.replace('.xlsx', '_ENHANCED_analysis_results.xlsx')
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main results
        summary_df.to_excel(writer, sheet_name='Investment_Analysis', index=False)
        
        # Investment tiers breakdown
        tier_breakdown = df.groupby('investment_tier').agg({
            'account name': 'count',
            'total_investment_score': ['mean', 'std'],
            'strategic_score': 'mean',
            'global_leader_count': 'mean'
        }).round(2)
        tier_breakdown.to_excel(writer, sheet_name='Investment_Tiers')
        
        # Territory analysis
        territory_analysis = df.groupby('territory').agg({
            'account name': 'count',
            'total_investment_score': 'mean',
            'global_leader_count': 'mean',
            'strategic_score': 'mean'
        }).round(2)
        territory_analysis.to_excel(writer, sheet_name='Territory_Analysis')
        
        # Top companies by category
        top_companies = df.nlargest(50, 'total_investment_score')[
            ['account name', 'territory', 'investment_tier', 'total_investment_score', 'strategic_score']
        ]
        top_companies.to_excel(writer, sheet_name='Top_50_Targets', index=False)
    
    print(f"Enhanced results exported to: {output_file}")
    return output_file

def main():
    """Main function to run the complete analysis"""
    print("=== ENHANCED BIOPHARMA INVESTMENT ANALYSIS ===")
    
    # Debug: Show current directory and files
    import os
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    print("\nFiles in current directory:")
    for file in os.listdir(current_dir):
        if file.endswith('.xlsx') or file.endswith('.xls'):
            print(f"  - {file}")
    
    print("\nMake sure your Excel file is in the same directory as this script")
    
    # Get file path from user
    file_path = input("Enter your Excel file name (e.g., 'biopharma_data.xlsx'): ")
    
    try:
        # Load and prepare data
        result = load_and_prepare_data(file_path)
        if result[0] is None:
            return
            
        df, classification_cols, strategic_col, df_original = result
        
        print(f"\n" + "="*60)
        print("RUNNING ENHANCED ANALYSIS SUITE")
        print("="*60)
        
        # Run all analyses
        print("\n1. Classification Distribution Analysis...")
        create_classification_distribution_heatmap(df, classification_cols)
        
        print("\n2. Investment Score Analysis...")
        create_investment_score_heatmap(df, classification_cols)
        
        print("\n3. Territory Analysis...")
        create_territory_analysis(df, classification_cols)
        
        print("\n4. Global Leaders Analysis...")
        create_global_leaders_analysis(df, classification_cols)
        
        print("\n5. Strategic Fit Analysis...")
        create_strategic_fit_analysis(df)
        
        # New enhanced analyses
        print("\n6. Investment Tier Analysis...")
        create_investment_tier_analysis(df, classification_cols)
        
        print("\n7. Competitive Landscape Analysis...")
        create_competitive_landscape_analysis(df, classification_cols)
        
        print("\n8. Portfolio Optimization Analysis...")
        create_portfolio_optimization_analysis(df, classification_cols)
        
        # Deduplication comparison if applicable
        if len(df_original) != len(df):
            print("\n9. Before/After Deduplication Comparison...")
            create_before_after_comparison(df_original, df)
        
        print("\n10. Actionable Insights Report...")
        create_actionable_insights_report(df, classification_cols, strategic_col)
        
        # Export enhanced results
        output_file = export_detailed_results(df, file_path)
        
        # Final summary statistics
        print("\n" + "="*60)
        print("FINAL SUMMARY STATISTICS")
        print("="*60)
        print(f"Total unique companies analyzed: {len(df)}")
        print(f"Average investment score: {df['total_investment_score'].mean():.2f}")
        print(f"Companies with global leadership: {len(df[df['global_leader_count'] >= 1])}")
        print(f"High strategic fit companies (≥4): {len(df[df['strategic_score'] >= 4])}")
        print(f"Tier 1 premium targets: {len(df[df['investment_tier'] == 'Tier 1: Premium Targets'])}")
        
        if len(df_original) != len(df):
            print(f"Duplicate entries cleaned: {len(df_original) - len(df)}")
        
        top_company = df.loc[df['total_investment_score'].idxmax()]
        print(f"\nTop investment target: {top_company['account name']}")
        print(f"  Score: {top_company['total_investment_score']:.1f}")
        print(f"  Territory: {top_company['territory']}")
        print(f"  Investment Tier: {top_company['investment_tier']}")
        
        print(f"\nEnhanced analysis complete! Results saved to:")
        print(f"  {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Make sure the file exists in the current directory.")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure your Excel file has the correct column names and format.")
        import traceback
        traceback.print_exc()

# Run the analysis
if __name__ == "__main__":
    main()
